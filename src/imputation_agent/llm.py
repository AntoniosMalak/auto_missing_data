
from __future__ import annotations
from typing import Optional, Literal
from dotenv import load_dotenv

Provider = Literal["openai", "ollama"]

def load_llm(provider: Provider="ollama", model: Optional[str]=None, temperature: float=0.0):
    """Return a LangChain-compatible chat model given a provider.
    - provider="openai" uses langchain_openai.ChatOpenAI (requires OPENAI_API_KEY)
    - provider="ollama" uses langchain_community.chat_models.ChatOllama (requires local Ollama)
    """
    load_dotenv()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model or "llama3.1:8b-instruct-q4_K_M", temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
