
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
    use_taxonomy: bool = typer.Option(True, help="Use taxonomy in agent mode"),
):
    result = plan_and_run(csv, out, provider=provider, model=model, temperature=temperature, use_taxonomy=use_taxonomy)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    app()
