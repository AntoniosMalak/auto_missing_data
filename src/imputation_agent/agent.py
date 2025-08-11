
from __future__ import annotations
import json
from typing import Optional, Literal
from dotenv import load_dotenv
from .llm import load_llm

def plan_and_run(csv_path: str, out_dir: str = "outputs",
                 provider: Literal["openai","ollama"]="ollama",
                 model: Optional[str]=None,
                 temperature: float=0.0,
                 use_taxonomy: bool = True) -> dict:
    load_dotenv()
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    from .tools import tool_profile, tool_run_pipeline

    profile_tool = Tool(name="profile_data", func=tool_profile, description="Profile CSV and return JSON.")
    run_tool = Tool(name="run_pipeline",
                    func=lambda path: tool_run_pipeline(path, out_dir, use_taxonomy=use_taxonomy),
                    description="Run pipeline; returns JSON with outputs.")

    llm = load_llm(provider=provider, model=model, temperature=temperature)
    system_msg = ("You are an imputation planning assistant. "
                  "First call profile_data, then run_pipeline. "
                  "Prefer simpler methods when scores are close.")

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
