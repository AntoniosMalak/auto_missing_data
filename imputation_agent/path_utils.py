from pathlib import Path

def normalize_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    return str(Path(p))
