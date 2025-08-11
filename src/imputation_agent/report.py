
from __future__ import annotations
import os, json, datetime as dt

def render_markdown(out_dir: str, report_json_path: str):
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    lines = []
    lines.append("# Imputation Report\n")
    ts = dt.datetime.fromtimestamp(report.get("timestamp", 0)).isoformat()
    lines.append(f"- Generated: {ts}")
    lines.append(f"- Rows: {report.get('n_rows')}")
    lines.append(f"- Cols: {report.get('n_cols')}\n")
    lines.append("## Selection\n```json")
    lines.append(json.dumps(report.get("selection", {}), indent=2))
    lines.append("```\n")
    md_path = os.path.join(out_dir, "imputation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
