"""Render the per-run findings document from annotations and mechanical tables.

The reference document states methodology and conclusions; this states what each
trajectory did. The relation between them is conclusion to record, so every
number in the reference should be walkable back to a run section here.

Sections are fixed and ordered so runs can be read side by side: what the run
was, what it changed, what it trained, what it verified, how it ended, and what
remained unresolved. The last of those is not an appendix — an annotator's
``unclear`` and the mechanical layer's unreachable cells are the honest edge of
what this corpus supports, and burying them would make the document read more
settled than it is.

Rows whose evidence pointer did not resolve are already gone by the time this
runs; it renders what survived the check.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_CAT_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def _cell(value: Any, width: int = 150) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return text[:width] + ("…" if len(text) > width else "")


def _yesno(value: Any) -> str:
    """``pd.NA`` is neither true nor false and raises if asked; say so."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return "是" if bool(value) else "否"


def _hours(sec: Any) -> str:
    if sec is None or sec is pd.NA or pd.isna(sec):
        return "—"
    return f"{float(sec) / 3600:.2f}h"


def _evidence(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "—"
    return ", ".join(f"i={e[0]}" for e in value if isinstance(e, (list, tuple)) and len(e) == 2)


def run_section(
    run_id: str, meta: dict[str, Any], tables: dict[str, pd.DataFrame],
    spans: pd.DataFrame, evals: pd.DataFrame,
) -> str:
    def mine(name: str) -> pd.DataFrame:
        df = tables.get(name)
        if df is None or not len(df) or "run_id" not in df:
            return pd.DataFrame()
        return df[df["run_id"] == run_id]

    out: list[str] = [f"\n## {run_id}\n"]

    out.append(
        "| agent | harness | benchmark | base model | 时长 | 最终分 |\n|---|---|---|---|---|---|\n"
        f"| {meta.get('model')} | {meta.get('harness')} | {meta.get('benchmark')} | "
        f"{meta.get('base')} | {_hours(meta.get('duration_s'))} | "
        f"{_cell(meta.get('score'), 12)} |\n"
    )

    ch = mine("changes")
    out.append(f"\n### 改动序列({len(ch)} 条)\n\n")
    if len(ch):
        out.append("| i | 类别 | 做了什么 | 证据 |\n|---|---|---|---|\n")
        for _, r in ch.sort_values("i").iterrows():
            out.append(
                f"| {_cell(r.get('i'), 8)} | {_cell(r.get('category'), 24)} | "
                f"{_cell(r.get('summary'))} | {_evidence(r.get('evidence'))} |\n"
            )
    else:
        out.append("*(无)*\n")

    tr = mine("trainings")
    out.append(f"\n### 训练序列({len(tr)} 段)\n\n")
    if len(tr):
        joined = tr.merge(
            spans[["run_id", "i", "sec", "kind", "end_reason"]], on=["run_id", "i"], how="left"
        )
        out.append("| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |\n|---|---|---|---|---|---|\n")
        for _, r in joined.sort_values("i").iterrows():
            out.append(
                f"| {_cell(r.get('i'), 8)} | {_cell(r.get('kind'), 8)} | {_hours(r.get('sec'))} | "
                f"{_cell(r.get('end_reason'), 12)} | **{_cell(r.get('tested_variable'), 10)}** | "
                f"{_cell(r.get('vs_previous'))} |\n"
            )
    else:
        out.append("*(无)*\n")

    ve = mine("verifications")
    out.append(f"\n### 验证序列({len(ve)} 次)\n\n")
    if len(ve):
        joined = ve.merge(
            evals[["run_id", "i", "tier", "limit", "got_signal"]],
            on=["run_id", "i"], how="left",
        )
        out.append("| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |\n|---|---|---|---|---|---|\n")
        for _, r in joined.sort_values("i").iterrows():
            judged = r.get("judges_changes")
            judged = ", ".join(map(str, judged)) if isinstance(judged, list) else "—"
            out.append(
                f"| {_cell(r.get('i'), 8)} | {_cell(r.get('tier'), 4)} | {_cell(r.get('limit'), 8)} | "
                f"{_yesno(r.get('got_signal'))} | {_cell(judged, 40)} | "
                f"{_cell(r.get('outcome_read'), 60)} |\n"
            )
    else:
        out.append("*(无)*\n")

    unclear = tr[tr.get("tested_variable") == "unclear"] if len(tr) else pd.DataFrame()
    no_signal = (
        ve.merge(evals[["run_id", "i", "got_signal"]], on=["run_id", "i"], how="left")
        if len(ve) else pd.DataFrame()
    )
    if len(no_signal):
        no_signal = no_signal[~no_signal["got_signal"].fillna(False).astype(bool)]
    out.append("\n### 异常与存疑\n\n")
    bits: list[str] = []
    if len(unclear):
        bits.append(f"- **{len(unclear)} 段训练的受测变量判不出**:i={list(unclear['i'])}\n")
    if len(no_signal):
        bits.append(f"- **{len(no_signal)} 次验证没有拿到信号**:i={list(no_signal['i'])}\n")
    for name, label in (
        ("proposed_category", "分类学缺口提案"),
        ("definition_defect", "定义缺陷"),
        ("boundary_case", "边界情形"),
    ):
        df = mine(name)
        if len(df):
            bits.append(f"- **{label} {len(df)} 条**\n")
            for _, r in df.iterrows():
                first = next(
                    (str(r[c]) for c in ("slug", "summary", "why", "note", "description")
                     if c in df.columns and isinstance(r.get(c), str) and r.get(c)),
                    "",
                )
                bits.append(f"  - {_cell(first, 200)}({_evidence(r.get('evidence'))})\n")
    out.append("".join(bits) if bits else "*(无)*\n")
    return "".join(out)


def render(
    annotations: dict[str, dict[str, Any]],
    tables: dict[str, pd.DataFrame],
    population: pd.DataFrame,
    spans: pd.DataFrame,
    evals: pd.DataFrame,
    header: str = "",
) -> str:
    """The whole document, runs sorted so families and benchmarks group."""
    meta = population.set_index("run_id").to_dict("index")
    order = sorted(
        annotations,
        key=lambda r: (
            meta.get(r, {}).get("harness") or "",
            meta.get(r, {}).get("benchmark") or "",
            meta.get(r, {}).get("model") or "",
            r,
        ),
    )
    parts = [header]
    for run_id in order:
        parts.append(run_section(run_id, meta.get(run_id, {}), tables, spans, evals))
    return "".join(parts)


__all__ = ["render", "run_section"]
