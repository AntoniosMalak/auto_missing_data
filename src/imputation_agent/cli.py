
import json, os
import typer
from rich import print
from .config import PipelineConfig
from .runner import run_pipeline
from .report import render_markdown
from .agent import plan_and_run

app = typer.Typer(add_completion=False, help="""
Imputation Agent CLI
- run: no-LLM deterministic pipeline
- plan-run: small agent shim (no external LLM by default), extendable to LangChain
""")


@app.command()
def run(
    csv: str = typer.Option(..., help="Path to input CSV"),
    out: str = typer.Option("outputs", help="Output directory"),
):
    cfg = PipelineConfig()
    out_csv, report_json, results, selection, profile = run_pipeline(csv, out, cfg)
    md = render_markdown(out, report_json)
    print(f"[bold green]Done.[/bold green] Imputed CSV: {out_csv}\nReport: {report_json}\nMarkdown: {md}")


@app.command("plan-run")
def plan_run(
    csv: str = typer.Option(..., help="Path to input CSV"),
    out: str = typer.Option("outputs", help="Output directory"),
):
    result = plan_and_run(csv, out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
