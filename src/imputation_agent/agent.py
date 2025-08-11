
from __future__ import annotations
import json
from typing import Optional, Literal
from dotenv import load_dotenv
from .llm import load_llm

def plan_and_run(csv_path: str, out_dir: str = "outputs",
                 provider: Literal["openai","ollama"]="ollama",
                 model: Optional[str]=None,
                 temperature: float=0.0) -> dict:
    """Agent execution:
    - Profiles the CSV
    - Lets an LLM (OpenAI or Ollama local) decide the plan
    - Executes the deterministic pipeline tool
    """
    load_dotenv()
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    from .tools import tool_profile, tool_run_pipeline

    profile_tool = Tool(
        name="profile_data",
        func=tool_profile,
        description="Profile the CSV at a given path and return a JSON string."
    )
    run_tool = Tool(
        name="run_pipeline",
        func=lambda path: tool_run_pipeline(path, out_dir),
        description="Run the imputation pipeline on a CSV path. Returns JSON with output paths."
    )

    llm = load_llm(provider=provider, model=model, temperature=temperature)

    system_msg = (
        "You are an imputation planning assistant. "
        "Given a CSV path, first call profile_data to understand types and missingness. "
        "Then call run_pipeline to execute. Prefer simpler methods when performance is similar."
    )

    agent = initialize_agent(
        tools=[profile_tool, run_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        system_message=system_msg,
    )

    res = agent.invoke({"input": f"Profile {csv_path} then run the pipeline saving outputs to {out_dir}."})
    return {"agent_output": res}
