#!/usr/bin/env python3
"""
LangGraph-based agent with Tavily Search Tool integration.
This agent uses ChatOpenAI (with OpenAI’s API) and binds a web search tool
(via TavilySearchResults) so that when the LLM decides it needs updated info,
it can perform a search.
"""

import os
from dotenv import load_dotenv
from typing import Annotated


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Ensure the required API keys are set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Please set your TAVILY_API_KEY in the .env file")
if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("Please set your LANGSMITH_API_KEY in the .env file")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # highlight-next-line
    name: str
    # highlight-next-line
    birthday: str



@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


tavily_search = TavilySearchResults(max_results=2)
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [tavily_search, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Initialize the state graph and the language model.
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with a MemorySaver for persistence.
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

def run_agent(initial_state: dict, config: dict = None) -> dict:
    """
    Invoke the compiled graph with the given initial state and configuration.

    Args:
        initial_state (dict): The starting state (e.g., messages).
        config (dict, optional): Additional configuration.

    Returns:
        dict: The final state after running the graph.
    """
    if config is None:
        config = {}
    return graph.invoke(initial_state, config=config)

def resume_human_assistance(human_response: str, config: dict = None) -> dict:
    """
    Resume the halted graph execution by providing a human assistance response.
    Returns the updated state after resuming.
    """
    if config is None:
        config = {}
    cmd = Command(resume={"data": human_response})
    return graph.invoke(cmd, config=config)

# --- Graph Visualization ---
if __name__ == "__main__":
    # This code uses IPython.display to render the graph image (useful in Jupyter/IPython)
    from IPython.display import Image, display

    try:
        # Generate the PNG image of the graph
        png_data = graph.get_graph().draw_mermaid_png()
        display(Image(png_data))
    except Exception as e:
        print("Graph visualization failed:", e)
