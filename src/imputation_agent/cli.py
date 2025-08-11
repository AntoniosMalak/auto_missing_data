
import json
import typer
from rich import print
from typing import Optional
from .config import PipelineConfig
from .runner import run_pipeline
from .report import render_markdown
from .agent import plan_and_run

app = typer.Typer(add_completion=False, help="""
Imputation Agent CLI
- run: deterministic pipeline
- plan-run: agent (OpenAI or Ollama) orchestrates tools
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
    provider: str = typer.Option("ollama", help="LLM provider: openai or ollama"),
    model: Optional[str] = typer.Option(None, help="Model name (e.g., 'gpt-4o-mini' or 'llama3.1:8b-instruct')"),
    temperature: float = typer.Option(0.0, help="LLM sampling temperature"),
):
    result = plan_and_run(csv, out, provider=provider, model=model, temperature=temperature)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
