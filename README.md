
# Imputation Agent (Python / LangChain / OpenAI or Ollama)

End-to-end pipeline:
1) Profile CSV
2) Try imputation techniques
3) Mask-and-score evaluation
4) Pick best per column
5) Impute full data
6) Generate artifacts

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Deterministic
```bash
python -m imputation_agent.cli run --csv path/to/your.csv --out outputs --use-taxonomy
```

## Agent with Ollama
```bash
# ollama pull llama3.1:8b-instruct
python -m imputation_agent.cli plan-run --csv path/to/your.csv --out outputs --provider ollama --model "llama3.1:8b-instruct" --use-taxonomy
```

## Agent with OpenAI
```bash
export OPENAI_API_KEY=sk-...
python -m imputation_agent.cli plan-run --csv path/to/your.csv --out outputs --provider openai --model gpt-4o-mini --use-taxonomy
```

### Taxonomy
Rules in `src/imputation_agent/taxonomy.py` choose candidates based on missing%/cardinality/size:
- Numeric: median → knn (if missing% < 40% & not huge) → mice (if rows < 300k)
- Categorical: most_frequent (+ constant for high-cardinality & high missing)
- Boolean: most_frequent
- Datetime: ffill/bfill
