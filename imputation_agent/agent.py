
from __future__ import annotations

import json, os
from typing import TypedDict, Optional, Literal, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from .llm import load_llm
from .tools import tool_profile, tool_run_pipeline, tool_llm_report

# Define the state for our graph
class AgentState(TypedDict):
    """Represents the state of our graph."""
    csv_path: str
    out_dir: str
    provider: str
    model: Optional[str]
    temperature: float
    # NOTE: these are JSON-encoded strings (to be compatible with current tools)
    profile: Optional[str]
    results: Optional[str]
    # selection is a JSON-serializable object (dict), not a plain string
    selection: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]

def profile_data_node(state: AgentState) -> AgentState:
    """Node to profile the data."""
    print("---Executing Node: Profile Data---")
    profile_json_str = tool_profile(state["csv_path"])  # returns JSON string
    return {"profile": profile_json_str}

def run_pipeline_node(state: AgentState) -> AgentState:
    """Node to run the imputation pipeline."""
    print("---Executing Node: Run Pipeline---")
    results_json_str = tool_run_pipeline(state["csv_path"], state["out_dir"])  # returns JSON string
    return {"results": results_json_str}

def _resolved_selection(state: AgentState) -> Dict[str, Any]:
    """Resolve the selection dict from state or from run_pipeline results."""
    if state.get("selection") and isinstance(state["selection"], dict):
        return state["selection"]
    # try to parse from results (which is a JSON string) -> { "selection": {...} }
    try:
        res = json.loads(state.get("results") or "{}")  # type: ignore[arg-type]
        sel = res.get("selection", {})
        if isinstance(sel, dict):
            return sel
    except Exception:
        pass
    return {}

def generate_report_node(state: AgentState) -> AgentState:
    """Node to generate the final LLM report (strict JSON)."""
    print("---Executing Node: Generate Report---")
    selection_obj = _resolved_selection(state)

    report_args = {
        "profile": state["profile"],          # json string
        "results": state["results"],          # json string
        # IMPORTANT: pass a JSON string for selection
        "selection": json.dumps(selection_obj, ensure_ascii=False),
        "provider": state["provider"],
        "model": state["model"],
        "temperature": state["temperature"]
    }
    final_report_str = tool_llm_report(**report_args)  # returns JSON string
    try:
        final_report = json.loads(final_report_str)
    except Exception:
        final_report = {"error": "LLM report was not valid JSON"}
    return {"report": final_report}

def plan_and_run(csv_path: str, out_dir: str = "outputs",
                 provider: Literal["openai","ollama"] = "ollama",
                 model: Optional[str] = None,
                 temperature: float = 0.0) -> dict:
    """Initializes and runs a LangGraph-based agent to perform profiling,
    imputation, and reporting in a reliable sequence, returning pure JSON."""
    load_dotenv()

    # Define the workflow graph
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph
    workflow.add_node("profile_data", profile_data_node)
    workflow.add_node("run_pipeline", run_pipeline_node)
    workflow.add_node("generate_report", generate_report_node)

    # Define the edges that control the flow
    workflow.set_entry_point("profile_data")
    workflow.add_edge("profile_data", "run_pipeline")
    workflow.add_edge("run_pipeline", "generate_report")
    workflow.add_edge("generate_report", END)

    # Compile the graph into a runnable app
    app = workflow.compile()

    # Define the initial inputs for the graph
    inputs = {
        "csv_path": csv_path,
        "out_dir": out_dir,
        "provider": provider,
        "model": model,
        "temperature": temperature
    }

    # Run the graph and get the final state
    final_state = app.invoke(inputs)
    # Return a JSON-friendly object directly (no nested dumped strings)
    return {"agent_output": final_state.get("report", {"error": "No report generated."})}
