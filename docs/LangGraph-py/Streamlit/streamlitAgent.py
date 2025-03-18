#!/usr/bin/env python3
"""
Streamlit user interface for the LangGraph-based chatbot.
"""

import os
import sys
# Add the parent directory to sys.path so that the Teams package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import nest_asyncio
from Teams.testAgent import run_agent, graph  # Import both run_agent and graph
from langchain_core.messages import HumanMessage

def main():
    # Set the page configuration
    st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")

    # Title of the app
    st.title("🤖 LangGraph Chatbot")

    # --- Display Graph Visualization ---
    try:
        # Generate the PNG image of the graph
        graph_image = graph.get_graph().draw_mermaid_png()
        st.image(graph_image, caption="Graph Visualization", width=100)
    except Exception as e:
        st.error(f"Graph visualization failed: {e}")

    # Initialize session state for chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the initial state for the agent using a HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "interview_started": False,
            "current_question": None,
            "current_question_id": None
        }

        with st.spinner("Thinking..."):
            try:
                # Ensure async compatibility with Streamlit
                nest_asyncio.apply()
                # Run the agent with the provided state and configuration
                result = run_agent(initial_state, config={"configurable": {"thread_id": "streamlit_chat"}})

                # Get the AI response (assumes it's the last message in the state's messages list)
                ai_response = result["messages"][-1].content

                # Add AI message to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
