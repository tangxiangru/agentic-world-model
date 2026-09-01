"""The per-run brief an annotating agent works from.

An agent that is handed a raw 900-event stream and asked what happened will
answer, and its answer will be unfalsifiable. The brief exists to make the
answerable part already answered: the mechanical layer has located every
training, every evaluation and every configuration access, so what remains for
the agent is the set of questions a parser cannot settle — which variable a
training was testing, which change an evaluation was judging, which category a
change belongs to.

Two properties matter more than the formatting.

*   **Every row carries its event index.** The agent is told where each fact came
    from and is required to answer in the same currency, so a claim can be walked
    back to the stream that produced it. A judgement with no ``(i, fragment)``
    pointer is rejected before anyone reads it.
*   **What the mechanical layer could not reach is stated, not hidden.** A codex
    run's configuration writes carry no content and its evaluation scores mostly
    cannot be linked to their launches; saying so in the brief is what stops the
    agent from inventing the missing half. The gaps are the agent's actual work.

The brief is text because the agent reads it, and the stream it points into is
the committed event file, so two runs of this produce the same input.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from awm import paths
from awm.traj import config_writes, eval_events, train_spans

#: Keep the brief inside a size an agent can hold alongside the stream it must
#: read. Runs above this are summarised the same way, just with the command
#: excerpts trimmed harder.
_CMD = 220


def _load(path: Path, module: Any) -> pd.DataFrame:
    return module.load(path) if path.exists() else module.empty()


def derived(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """The three mechanical tables, loaded once for a whole batch."""
    d = Path(root) if root is not None else paths.data_root() / "traj/derived"
    return {
        "spans": _load(d / "cc_train_spans_v1.parquet", train_spans),
        "evals": _load(d / "ptb_eval_events_v1.parquet", eval_events),
        "configs": _load(d / "ptb_config_writes_v1.parquet", config_writes),
    }


def _rows(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    return df[df["run_id"] == run_id].sort_values("i") if len(df) else df


def _cell(value: Any, width: int = _CMD) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = str(value).replace("\n", " ⏎ ").replace("|", "\\|")
    return text[:width] + ("…" if len(text) > width else "")


def _hours(seconds: Any) -> str:
    if seconds is None or seconds is pd.NA or pd.isna(seconds):
        return "—"
    return f"{float(seconds) / 3600:.2f}h"


def brief(run_id: str, tables: dict[str, pd.DataFrame], meta: dict[str, Any]) -> str:
    """One run's mechanical skeleton, as the agent receives it."""
    spans = _rows(tables["spans"], run_id)
    evals = _rows(tables["evals"], run_id)
    configs = _rows(tables["configs"], run_id)

    out: list[str] = []
    out.append(f"# {run_id}\n")
    out.append(
        "| agent | harness | benchmark | base model | 时长 | 最终分 |\n|---|---|---|---|---|---|\n"
        f"| {meta.get('model')} | {meta.get('harness')} | {meta.get('benchmark')} | "
        f"{meta.get('base')} | {_hours(meta.get('duration_s'))} | {meta.get('score')} |\n"
    )

    out.append("\n## 训练(机械抽取,已完整)\n")
    if len(spans):
        out.append("| i | 产物 | 类型 | 启动 | 时长 | 结局 | 命令 |\n|---|---|---|---|---|---|---|\n")
        for _, r in spans.iterrows():
            out.append(
                f"| {r['i']} | {_cell(r['out_dir'], 40)} | {r['kind']} | {r['mode']} | "
                f"{_hours(r['sec'])} | {r['end_reason']} | {_cell(r['command'])} |\n"
            )
    else:
        out.append("*(未检出训练)*\n")

    out.append("\n## 评测(机械抽取)\n")
    if len(evals):
        out.append(
            "| i | 被评模型 | limit | 档位 | 启动 | 拿到分数 | 分数 | 通道 | 命令 |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for _, r in evals.iterrows():
            out.append(
                f"| {r['i']} | {_cell(r['model_path'], 40)} | {_cell(r['limit'], 8)} | "
                f"{_cell(r['tier'], 4)} | {r['mode']} | {'是' if r['got_signal'] else '否'} | "
                f"{_cell(r['accuracy'], 8)} | {_cell(r['signal_via'], 10)} | {_cell(r['command'])} |\n"
            )
    else:
        out.append("*(未检出评测)*\n")

    out.append("\n## `generation_config.json` 访问(机械抽取,仅定位)\n")
    if len(configs):
        out.append("| i | 路径 | 访问 | 形式 | 有内容 | 命令/内容 |\n|---|---|---|---|---|---|\n")
        for _, r in configs.iterrows():
            payload = r["content"] if r["content_available"] else r["command"]
            out.append(
                f"| {r['i']} | {_cell(r['path'], 46)} | {r['access']} | {r['form']} | "
                f"{'是' if r['content_available'] else '否'} | {_cell(payload)} |\n"
            )
    else:
        out.append("*(未检出访问)*\n")

    out.append("\n## 机械层在这条 run 上够不到的\n")
    limits: list[str] = []
    n_unlinked = int((~evals["got_signal"].fillna(False)).sum()) if len(evals) else 0
    if n_unlinked:
        limits.append(
            f"- **{n_unlinked} / {len(evals)} 次评测没能关联到分数。** 这是「追不到」,不是「agent 没看到」。"
            "请在流里找它实际是怎么把分数读回去的(自写分析脚本、读 inspect_ai 日志目录、或确实没读)。"
        )
    n_noc = int((~configs["content_available"].fillna(False)).sum()) if len(configs) else 0
    if n_noc:
        limits.append(
            f"- **{n_noc} / {len(configs)} 次 config 访问拿不到内容。** 字段级差异要你读那段命令得出:"
            "特别注意**整份重写**会顺手删掉原有字段(如 `max_new_tokens`),看起来像只改了一项。"
        )
    n_unclear = int((spans["end_reason"] == "run_end").sum()) if len(spans) else 0
    if n_unclear:
        limits.append(
            f"- **{n_unclear} 次训练启动后再没被提及**,结束时刻取的是 run 结尾,是上界。"
        )
    n_open = int(spans["end_reason"].isin(["consumed", "last_seen"]).sum()) if len(spans) else 0
    if n_open:
        limits.append(
            f"- **{n_open} 次训练的结束时刻靠「产物被消费」推断,崩溃的训练在这里两个方向都可能错**:"
            "产物被中途读取会让时长偏短,崩溃后 GPU 空转会让时长偏长。请在流里找 traceback / "
            "`pkill` / 显存归零,给出真实结局。"
        )
    out.append("".join(limits) if limits else "*(无)*\n")
    return "".join(out)


def population(index: pd.DataFrame, models: set[str], benchmarks: set[str]) -> pd.DataFrame:
    """The in-scope runs, with the base model parsed out of the run id."""
    import re

    known = ("gsm8k", "bfcl", "aime2025", "humaneval", "gpqamain", "healthbench",
             "arenahardwriting")

    def base(run_id: str) -> str | None:
        for b in known:
            m = re.search(rf"__{b}_(.+?)_\d+$", run_id)
            if m:
                return m.group(1)
        return None

    df = index[
        (index["source"] == "posttrainbench")
        & index["model"].isin(models)
        & index["benchmark"].isin(benchmarks)
    ].copy()
    df["base"] = df["run_id"].map(base)
    return df


def has_clock(run_id: str, events_root: Path | None = None) -> bool:
    """Whether this run's stream carries timestamps at all."""
    root = Path(events_root) if events_root is not None else paths.events_dir("posttrainbench")
    path = root / f"{run_id}.jsonl.gz"
    if not path.exists():
        return False
    with gzip.open(path, "rt") as fh:
        return any(json.loads(line).get("ts") for line in fh if line.strip())


__all__ = ["brief", "derived", "has_clock", "population"]
