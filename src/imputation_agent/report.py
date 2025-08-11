
from __future__ import annotations
import os, json, datetime as dt
from typing import Dict, Any


def render_markdown(out_dir: str, report_json_path: str):
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    lines = []
    lines.append(f"# Imputation Report")
    lines.append("")
    ts = dt.datetime.fromtimestamp(report.get("timestamp", 0))
    lines.append(f"- Generated: {ts.isoformat()}")
    lines.append(f"- Rows: {report.get('n_rows')}")
    lines.append(f"- Cols: {report.get('n_cols')}")
    lines.append("")
    lines.append("## Methods Tried")
    lines.append("```json")
    lines.append(json.dumps(report.get("methods_tried", {}), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Selection (per column)")
    lines.append("```json")
    lines.append(json.dumps(report.get("selection", {}), indent=2))
    lines.append("```")

    md_path = os.path.join(out_dir, "imputation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
