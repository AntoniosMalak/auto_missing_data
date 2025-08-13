
import json, os
import typer
from rich import print
from typing import Optional
from .config import PipelineConfig
from .runner import run_pipeline
from .report import render_markdown
from .agent import plan_and_run
from .llm_report import generate_llm_json_report

app = typer.Typer(add_completion=False, help="""
Imputation Agent CLI
- run: deterministic pipeline (optionally taxonomy) + optional LLM JSON report
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
        # Compose inputs for LLM report
        with open(os.path.join(out, 'all_methods_report.json'), 'r', encoding='utf-8') as f:
            allm = json.load(f)
        with open(report_json, 'r', encoding='utf-8') as f:
            sel = json.load(f).get("selection", {})
        prof = {"n_rows": profile.n_rows, "n_cols": profile.n_cols,
                "columns": [{"name": c.name, "dtype": c.dtype, "missing_rate": c.missing_rate} for c in profile.columns]}
        llm_json = generate_llm_json_report(prof, allm, sel, provider=provider, model=model, temperature=temperature)
        out_path = os.path.join(out, 'llm_report.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(llm_json, f, ensure_ascii=False, indent=2)
        print(f"[bold cyan]LLM JSON report:[/bold cyan] {out_path}")

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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)



# -------- Anomaly subcommand --------
@app.command("anomaly")
def anomaly_cmd(
    input_csv: str = typer.Option(..., help="Input CSV file"),
    out_dir: str = typer.Option("./out_anomaly", help="Output directory"),
    seeds: str = typer.Option("1,2,3", help="Comma-separated random seeds"),
):
    """
    Runs per-column anomaly detection model selection and outputs a JSON report.
    """
    import os, json
    import pandas as pd
    from .profiling import infer_profile, parse_datetimes_inplace
    from .anomaly_runner import try_anomaly_methods, pick_best_anomaly_per_column, apply_anomaly_treatment

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    parse_datetimes_inplace(df)
    profile = infer_profile(df)

    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    methods_map = {
        "numeric": ["zscore","iqr","isolation_forest","lof"],
        "categorical": ["rare","unexpected"],
        "datetime": ["zscore"],
        "boolean": ["none"],
    }

    # Evaluate & select
    results = try_anomaly_methods(df, profile, methods_map, seeds_list)
    dtype_map = {cp.name: cp.dtype for cp in profile.columns}
    selection = pick_best_anomaly_per_column(results, dtype_map)

    # Apply treatment (set anomalies to NaN) and save
    df_treated, counts = apply_anomaly_treatment(df, selection)
    out_csv = os.path.join(out_dir, "treated_with_anomaly.csv")
    df_treated.to_csv(out_csv, index=False)

    report = {
        "selection": selection,
        "counts": counts,
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
    }
    with open(os.path.join(out_dir, "anomaly_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    typer.echo(f"Wrote: {out_csv}")
    typer.echo(f"Wrote: {os.path.join(out_dir,'anomaly_report.json')}")


@app.command("run-with-anomaly")
def run_with_anomaly(
    input_csv: str = typer.Option(..., help="Input CSV file"),
    out_dir: str = typer.Option("./out_full", help="Output directory"),
    seeds: str = typer.Option("1,2,3", help="Comma-separated random seeds"),
    taxonomy: Optional[str] = typer.Option(None, help="Optional taxonomy JSON to include in the LLM report."),
    llm: bool = typer.Option(False, help="Also produce LLM JSON report (same as --llm in run)."),
):
    """
    1) Detect anomalies per column, select best detector by F1 on synthetic anomalies, mark anomalies as NaN
    2) Run the existing imputation model-selection pipeline on the treated data
    """
    import os, json, shutil, pandas as pd
    from .profiling import infer_profile, parse_datetimes_inplace
    from .anomaly_runner import try_anomaly_methods, pick_best_anomaly_per_column, apply_anomaly_treatment
    from .config import PipelineConfig
    from .runner import run_pipeline
    from .llm_report import generate_llm_json_report

    os.makedirs(out_dir, exist_ok=True)
    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    raw_df = pd.read_csv(input_csv)
    parse_datetimes_inplace(raw_df)
    profile = infer_profile(raw_df)
    methods_map = {
        "numeric": ["zscore","iqr","isolation_forest","lof"],
        "categorical": ["rare","unexpected"],
        "datetime": ["zscore"],
        "boolean": ["none"],
    }
    # anomaly step
    anom_results = try_anomaly_methods(raw_df, profile, methods_map, seeds_list)
    dtype_map = {cp.name: cp.dtype for cp in profile.columns}
    anom_sel = pick_best_anomaly_per_column(anom_results, dtype_map)
    treated_df, counts = apply_anomaly_treatment(raw_df, anom_sel)
    treated_csv = os.path.join(out_dir, "step1_treated.csv")
    treated_df.to_csv(treated_csv, index=False)
    with open(os.path.join(out_dir, "step1_anomaly_selection.json"), "w", encoding="utf-8") as f:
        json.dump({"selection": anom_sel, "counts": counts}, f, ensure_ascii=False, indent=2)

    # imputation step
    cfg = PipelineConfig(input_csv=treated_csv, out_dir=os.path.join(out_dir, "imputation"))
    run_pipeline(cfg)

    # optional LLM report
    if llm:
        prof = {"n_rows": profile.n_rows, "n_cols": profile.n_cols}
        with open(os.path.join(out_dir, "imputation", "imputation_report.json"), "r", encoding="utf-8") as f:
            sel = json.load(f)
        allm = json.loads(taxonomy) if taxonomy else {}
        llm_json = generate_llm_json_report(prof, allm, sel)
        with open(os.path.join(out_dir, "llm_report.json"), "w", encoding="utf-8") as f:
            json.dump(llm_json, f, ensure_ascii=False, indent=2)

    print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()