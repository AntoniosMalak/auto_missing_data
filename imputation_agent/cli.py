
import json, os
import typer
from rich import print
from typing import Optional
from .config import PipelineConfig
from .runner import run_pipeline
from .report import render_markdown
from .agent import plan_and_run
from .llm_report import generate_llm_json_report
from .profiling import infer_profile, parse_datetimes_inplace
from .anomaly_runner import try_anomaly_methods, pick_best_anomaly_per_column, apply_anomaly_treatment

app = typer.Typer(add_completion=False, help="""
Imputation Agent CLI
- run: deterministic pipeline (optionally taxonomy) + optional LLM JSON report
- run-with-anomaly: anomaly selection -> treat -> imputation -> optional LLM JSON report
- plan-run: agent (OpenAI or Ollama) orchestrates tools and returns tool outputs
""")

@app.command()
def run(
    csv: str = typer.Option(..., help="Path to input CSV"),
    out: str = typer.Option("outputs", help="Output directory"),
    llm_report: bool = typer.Option(False, help="Also generate LLM JSON report (requires provider/model)"),
    provider: str = typer.Option("ollama", help="LLM provider: openai or ollama"),
    model: Optional[str] = typer.Option(None, help="Model name for provider"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
):
    cfg = PipelineConfig()
    out_csv, report_json, results, selection, profile = run_pipeline(csv, out, cfg)
    md = render_markdown(out, report_json)
    print(f"[bold green]Done.[/bold green] Imputed CSV: {out_csv}\nReport: {report_json}\nAll methods: {os.path.join(out, 'all_methods_report.json')}\nMarkdown: {md}")
    if llm_report:
        with open(os.path.join(out, 'all_methods_report.json'), 'r', encoding='utf-8') as f:
            allm = json.load(f)
        with open(report_json, 'r', encoding='utf-8') as f:
            sel = json.load(f).get("selection", {})
        prof = {"n_rows": profile.n_rows, "n_cols": profile.n_cols,
                "columns": [{"name": c.name, "dtype": c.dtype, "missing_rate": c.missing_rate} for c in profile.columns]}
        # add artifacts
        allm.setdefault("artifacts", {
            "imputed_csv": os.path.join(out, "imputed.csv"),
            "all_methods_report": os.path.join(out, "all_methods_report.json"),
            "report_json": os.path.join(out, "imputation_report.json"),
        })
        llm_json = generate_llm_json_report(prof, allm, sel, provider=provider, model=model, temperature=temperature)
        out_path = os.path.join(out, 'llm_report.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(llm_json, f, ensure_ascii=False, indent=2)
        print(f"[bold cyan]LLM JSON report:[/bold cyan] {out_path}")

@app.command("run-with-anomaly")
def run_with_anomaly(
    csv: str = typer.Option(..., help="Path to input CSV"),
    out: str = typer.Option("outputs_anomaly", help="Output directory"),
    seeds: str = typer.Option("1,2,3", help="Comma-separated random seeds for anomaly evaluation"),
    llm_report: bool = typer.Option(False, help="Also generate LLM JSON report (requires provider/model)"),
    provider: str = typer.Option("ollama", help="LLM provider: openai or ollama"),
    model: Optional[str] = typer.Option(None, help="Model name for provider"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
):
    os.makedirs(out, exist_ok=True)
    import pandas as pd
    df_raw = pd.read_csv(csv)
    parse_datetimes_inplace(df_raw)
    profile = infer_profile(df_raw)

    seeds_list = [int(s.strip()) for s in seeds.split(',') if s.strip()]
    methods_map = {
        "numeric": ["zscore","iqr","isolation_forest","lof"],
        "categorical": ["rare","unexpected"],
        "datetime": ["zscore"],
        "boolean": ["none"],
    }

    # 1) Evaluate anomaly detectors per column
    anom_results = try_anomaly_methods(df_raw, profile, methods_map, seeds_list)
    dtype_map = {cp.name: cp.dtype for cp in profile.columns}
    anom_sel = pick_best_anomaly_per_column(anom_results, dtype_map)

    # 2) Apply treatment (set detected anomalies to NaN)
    treated_df, anom_counts = apply_anomaly_treatment(df_raw, {c: {**v, "dtype": dtype_map.get(c, "numeric")} for c, v in anom_sel.items()})
    treated_csv = os.path.join(out, "step1_treated.csv")
    treated_df.to_csv(treated_csv, index=False)
    anomaly_report_path = os.path.join(out, "step1_anomaly_report.json")
    with open(anomaly_report_path, 'w', encoding='utf-8') as f:
        json.dump({"selection": anom_sel, "counts": anom_counts}, f, ensure_ascii=False, indent=2)

    # 3) Imputation on treated data
    cfg = PipelineConfig()
    out_csv, report_json, results, selection, profile2 = run_pipeline(treated_csv, os.path.join(out, "imputation"), cfg)

    # 4) Optional: LLM JSON report merging anomaly + imputation
    if llm_report:
        # Build a combined results object that includes anomaly results for the LLM
        with open(os.path.join(out, "imputation", "all_methods_report.json"), 'r', encoding='utf-8') as f:
            allm = json.load(f)

        # Attach anomaly info & artifacts so the LLM can reference them
        allm["anomaly_results"] = anom_results
        allm["anomaly_selection"] = anom_sel
        allm.setdefault("artifacts", {
            "treated_csv": treated_csv,
            "anomaly_report": anomaly_report_path,
            "imputed_csv": os.path.join(out, "imputation", "imputed.csv"),
            "all_methods_report": os.path.join(out, "imputation", "all_methods_report.json"),
            "report_json": os.path.join(out, "imputation", "imputation_report.json"),
        })

        # Profile for the LLM
        prof = {
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "columns": [{"name": c.name, "dtype": c.dtype, "missing_rate": c.missing_rate} for c in profile.columns]
        }
        llm_json = generate_llm_json_report(prof, allm, selection, provider=provider, model=model, temperature=temperature)
        out_path = os.path.join(out, 'llm_report.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(llm_json, f, ensure_ascii=False, indent=2)
        print(f"[bold cyan]LLM JSON report (anomaly+impute):[/bold cyan] {out_path}")

    print(f"[bold green]Done.[/bold green] Treated CSV: {treated_csv}\nAnomaly report: {anomaly_report_path}\nImputation dir: {os.path.join(out, 'imputation')}")

@app.command("plan-run")
def plan_run(
    csv: str = typer.Option(..., help="Path to input CSV"),
    out: str = typer.Option("outputs", help="Output directory"),
    provider: str = typer.Option("ollama", help="LLM provider: openai or ollama"),
    model: Optional[str] = typer.Option(None, help="Model name (e.g., 'gpt-4o-mini' or 'llama3.1:8b-instruct')"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
):
    result = plan_and_run(csv, out, provider=provider, model=model, temperature=temperature)
    out_path = os.path.join(out, 'llm_report.json')
    os.makedirs(out, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[bold cyan]LLM JSON report:[/bold cyan] {out_path}")

if __name__ == "__main__":
    app()
