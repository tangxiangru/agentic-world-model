"""Synthetic production-container acceptance; never a scored PTB experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from awm.exp_protocol import decisions, lock, schema

from . import isolation, sidecar
from .backends import Budget
from .review import default_skill_dir, make_brief


def accept(output: Path, model: str, effort: str) -> dict:
    # The production private archive intentionally excludes the scientist client.
    # The synthetic harness imports that client from a separately bound public archive.
    import awm
    public = os.environ.get("AWM_PUBLIC_CHECKOUT")
    if public:
        awm.__path__.append(str(Path(public) / "awm"))
    from awm import wma_client

    session = output / "session"
    cards = session / "memory/cards"
    cards.mkdir(parents=True)
    script = session / "diagnostic.py"
    script.write_text("print(1)\n")
    card = {
        "schema_version": schema.CARD_SCHEMA, "card_id": "exp-01",
        "situation": {"remaining_h": 1, "trigger": "Synthetic runtime acceptance only"},
        "problem": {"statement": "Verify a trivial diagnostic command; no training or benchmark"},
        "hypothesis": {"claim": "The supplied Python script prints one; no model-quality claim"},
        "setup": {"method": {"family": "other"}, "parent_checkpoint": {"path": "none"},
                  "command": {"argv": [sys.executable, str(script)], "cwd": str(session),
                              "script": str(script), "env": {}},
                  "data": [], "budget": {"planned_h": 0.01}},
        "evaluation": {"metric": "diagnostic_success", "protocol": {"n": 1},
                       "comparator": {"ref": "diagnostic", "value": 0}},
        "result": {"execution": "not_run"}, "conclusion": {},
    }
    card_path = cards / "exp-01.yaml"
    schema.dump_card(card_path, card)
    lock.write_lock(card_path, card, {"synthetic_runtime_fixture": True})
    budget = Budget(cpu_min=2, gpu_min=0, wall_min=6, max_turns=20)
    brief = make_brief(session, "exp-01", mode="online", budget=budget,
                       model=model, effort=effort, skill_dir=default_skill_dir())
    brief.verdict_path = output / "canary-result.json"
    secret = output / "forbidden-canary.txt"
    secret.write_text("SYNTHETIC_FORBIDDEN_CANARY")
    canary = f"""import pathlib,socket,subprocess,sys
try:
 pathlib.Path({str(secret)!r}).read_text()
except PermissionError:
 pass
else:
 raise AssertionError('outside read succeeded')
for family in (socket.AF_INET,socket.AF_INET6,socket.AF_UNIX):
 try:
  socket.socket(family)
 except PermissionError:
  continue
 raise AssertionError('socket succeeded')
pathlib.Path('scratch-ok').write_text('ok')
subprocess.run([sys.executable,'-I','-c','print(1)'],check=True)
print('CANARIES_OK')
"""
    with isolation.isolated_tools(brief, ["claude"], "claude") as configured:
        result = isolation.run_probe(configured.policy,
                                    shlex.join([sys.executable, "-I", "-c", canary]))
        if result["returncode"] or "CANARIES_OK" not in result["stdout"]:
            raise RuntimeError(f"OS canaries failed: {result}")
        isolation.broker_call(configured.policy, "write_result", {"result": {"canaries": True}})
        limits = configured.policy["limits"]

    proposal_file = decisions.create_proposal(session)
    proposal = json.loads(proposal_file.read_text())
    proposal["situation"].update(remaining_h=1, incumbent="none; synthetic smoke",
                                   evidence=["A local diagnostic.py prints one; no benchmark data"])
    for candidate in proposal["candidates"]:
        candidate.update(hypothesis="a tiny diagnostic can check the runtime",
                         parent_checkpoint="none", change="print a literal without training",
                         train_h=0, eval_h=0.01, cost_basis="a trivial Python command",
                         evidence=["synthetic acceptance fixture, not scientific evidence"],
                         uncertainty="no model-quality outcome is being asserted",
                         decision_test="whether the requested diagnostic completes")
    proposal["candidates"][1].update(
        hypothesis="a JSON round-trip tests parsing as well as interpreter startup",
        change="parse a fixed synthetic JSON object and serialize it again", eval_h=0.02,
        evidence=["stdlib json.loads/json.dumps are available; no dataset or trained model is involved"],
    )
    proposal_file.write_text(json.dumps(proposal))
    config = sidecar.Config(session_dir=session, skill_dir=default_skill_dir(),
                           history_dir=None, backend="claude", model=model, effort=effort,
                           budget=budget, jobs=1, private_output_dir=output / "private")
    sidecar.run(config, once=True)
    worker = threading.Thread(target=sidecar.run, args=(config,),
                              kwargs={"poll_seconds": 0.1}, daemon=True)
    worker.start()
    try:
        joint = wma_client.compare_and_wait(session, proposal["decision_id"], timeout_min=7)
        if joint["state"] != "completed":
            raise RuntimeError(f"real comparison failed: {joint}")
        reviewed = wma_client.review_and_wait(session, "exp-01", timeout_min=7)
        if reviewed["state"] != "delivered":
            raise RuntimeError(f"real blocking review failed: {reviewed}")
        verdict = json.loads((cards / "exp-01.verdict.json").read_text())
        comparison = json.loads(Path(joint["comparison_path"]).read_text())
        for value in (verdict, comparison):
            if value.get("model") != model or value.get("isolation", {}).get("method") != "landlock-seccomp-mcp-v1":
                raise RuntimeError("model identity or enforced broker evidence is missing")
        if verdict["request_id"] != reviewed["request_id"]:
            raise RuntimeError("blocking review did not return the matching request")
        inventories = []
        for transcript in (output / "private").rglob("*.transcript*.jsonl"):
            for line in transcript.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "system" and event.get("subtype") == "init":
                    tools = event.get("tools")
                    if not isinstance(tools, list) or not tools:
                        raise RuntimeError("real CLI tool inventory is missing")
                    if any(not isinstance(tool, str) or not tool.startswith("mcp__wma_probe__")
                           for tool in tools):
                        raise RuntimeError(f"unexpected native or foreign tools in isolated CLI: {tools}")
                    inventories.append({"transcript": str(transcript), "tools": tools})
        if len(inventories) < 2:
            raise RuntimeError("comparison and review tool inventories were not both retained")
        return {"passed": True, "synthetic": True, "scientific_completion": "not_assessed",
                "model": model, "effort": effort, "os_canaries": "passed",
                "comparison": joint, "review": reviewed, "probe_limits": limits,
                "tool_inventories": inventories,
                "private_sha": os.environ.get("POST_TRAIN_BENCH_WMA_CHECKOUT_SHA"),
                "public_sha": os.environ.get("AWM_CHECKOUT_SHA"),
                "uid": os.getuid(), "kernel": platform.release(),
                "node": os.environ.get("SLURMD_NODENAME"),
                "job_id": os.environ.get("SLURM_JOB_ID")}
    finally:
        (session / ".wma/stop").touch()
        worker.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="high")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = accept(args.out, args.model, args.effort)
    except (ValueError, OSError, RuntimeError) as exc:
        result = {"passed": False, "synthetic": True, "error": str(exc)}
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.out / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
