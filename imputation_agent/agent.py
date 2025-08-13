from __future__ import annotations
import json, os
from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from .llm import load_llm
from .tools import tool_profile, tool_run_pipeline, tool_llm_report

# Define the state for our graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        csv_path: The path to the input CSV file.
        out_dir: The directory to save output files.
        provider: The LLM provider (e.g., "ollama").
        model: The specific LLM model to use.
        temperature: The temperature for the LLM.
        profile: The JSON output from the profiling step.
        results: The JSON output from the pipeline run step.
        selection: A placeholder for method selection (can be enhanced later).
        report: The final JSON report from the LLM.
    """
    csv_path: str
    out_dir: str
    provider: str
    model: Optional[str]
    temperature: float
    profile: Optional[dict]
    results: Optional[dict]
    selection: Optional[str] # Or a more complex type if needed
    report: Optional[dict]

# --- Node Functions for the Graph ---

def profile_data_node(state: AgentState) -> AgentState:
    """Node to profile the data."""
    print("---Executing Node: Profile Data---")
    profile_json = tool_profile(state["csv_path"])
    return {"profile": profile_json}

def run_pipeline_node(state: AgentState) -> AgentState:
    """Node to run the imputation pipeline."""
    print("---Executing Node: Run Pipeline---")
    results_json = tool_run_pipeline(state["csv_path"], state["out_dir"])
    return {"results": results_json}

def generate_report_node(state: AgentState) -> AgentState:
    """Node to generate the final LLM report."""
    print("---Executing Node: Generate Report---")
    report_args = {
        "profile": state["profile"],
        "results": state["results"],
        "selection": json.dumps(state.get("selection", "The agent selected the best method based on the provided data.")),
        "provider": state["provider"],
        "model": state["model"],
        "temperature": state["temperature"]
    }
    final_report = tool_llm_report(**report_args)
    return {"report": final_report}

# --- Main Function ---

def plan_and_run(csv_path: str, out_dir: str = "outputs",
                 provider: Literal["openai","ollama"]="ollama",
                 model: Optional[str]=None,
                 temperature: float=0.0) -> dict:
    """
    Initializes and runs a LangGraph-based agent to perform profiling,
    imputation, and reporting in a reliable sequence.
    """
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

    return {"agent_output": json.dumps(final_state.get("report", {"error": "No report generated."}), ensure_ascii=False, indent=2)}