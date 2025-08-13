
from __future__ import annotations
from typing import Optional, Literal
from dotenv import load_dotenv

Provider = Literal["openai", "ollama"]

def load_llm(provider: Provider = "ollama", model: Optional[str] = None, temperature: float = 0.0):
    load_dotenv()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1:8b", temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
