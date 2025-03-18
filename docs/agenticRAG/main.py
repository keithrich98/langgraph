from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Sequence, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import os
import streamlit as st
import openai
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate the LangChain OpenAI LLM for non–function-calling interactions
llm = OpenAI(temperature=0.7, model_name="o3-mini", openai_api_key=openai_api_key)

# Create Dummy Data
research_texts = [
    "Research Report: Results of a New AI Model Improving Image Recognition Accuracy to 98%",
    "Academic Paper Summary: Why Transformers Became the Mainstream Architecture in Natural Language Processing",
    "Latest Trends in Machine Learning Methods Using Quantum Computing"
]

development_texts = [
    "Project A: UI Design Completed, API Integration in Progress",
    "Project B: Testing New Feature X, Bug Fixes Needed",
    "Product Y: In the Performance Optimization Stage Before Release"
]

# Text splitting settings
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Generate Document objects from text
research_docs = splitter.create_documents(research_texts)
development_docs = splitter.create_documents(development_texts)

# Create vector stores using embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

research_vectorstore = Chroma.from_documents(
    documents=research_docs,
    embedding=embeddings,
    collection_name="research_collection"
)

development_vectorstore = Chroma.from_documents(
    documents=development_docs,
    embedding=embeddings,
    collection_name="development_collection"
)

research_retriever = research_vectorstore.as_retriever()
development_retriever = development_vectorstore.as_retriever()

# Create retriever tools for function calling
research_tool = create_retriever_tool(
    research_retriever,
    "research_db_tool",
    "Search information from the research database."
)

development_tool = create_retriever_tool(
    development_retriever,
    "development_db_tool",
    "Search information from the development database."
)

# Combine tools into a list
tools = [research_tool, development_tool]

AgentState = Dict[str, Any]

def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]

    # Extract the user query from the incoming messages
    if isinstance(messages[0], tuple):
        user_message = messages[0][1]
    else:
        user_message = messages[0].content

    # Basic prompt to decide which tool to call
    prompt = (
        f"Given this user question: '{user_message}', decide if it pertains to research or development. "
        "If so, return a function call with the appropriate tool and a query parameter. Otherwise, just answer directly."
    )

    # Define the function schemas for OpenAI's function calling
    functions = [
        {
            "name": "research_db_tool",
            "description": "Search information from the research database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for research topics."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "development_db_tool",
            "description": "Search information from the development database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for development topics."
                    }
                },
                "required": ["query"]
            }
        }
    ]

    # Call OpenAI ChatCompletion with function calling enabled
    response = openai.ChatCompletion.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call="auto",
        temperature=0.7,
        max_tokens=1024,
        api_key=openai_api_key
    )

    message = response["choices"][0]["message"]

    # If a function call was returned, parse it and call the appropriate tool
    if "function_call" in message:
        func_call = message["function_call"]
        function_name = func_call["name"]
        arguments = func_call.get("arguments", "{}")
        try:
            args = json.loads(arguments)
        except Exception as e:
            args = {"query": arguments}
        query = args.get("query", "")
        if function_name == "research_db_tool":
            results = research_retriever.invoke(query)
            return {"messages": [AIMessage(content=f'Action: research_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
        elif function_name == "development_db_tool":
            results = development_retriever.invoke(query)
            return {"messages": [AIMessage(content=f'Action: development_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
    else:
        # If no function call, return the direct answer from the LLM
        return {"messages": [AIMessage(content=message["content"])]}

def simple_grade_documents(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print("Evaluating message:", last_message.content)

    # Check if retrieved documents are present in the response
    if "Results: [Document" in last_message.content:
        print("---DOCS FOUND, GO TO GENERATE---")
        return "generate"
    else:
        print("---NO DOCS FOUND, TRY REWRITE---")
        return "rewrite"

def generate(state: AgentState):
    print("---GENERATE FINAL ANSWER---")
    messages = state["messages"]
    question = messages[0].content if isinstance(messages[0], tuple) else messages[0].content
    last_message = messages[-1]

    # Extract document information from the results
    docs = ""
    if "Results: [" in last_message.content:
        results_start = last_message.content.find("Results: [")
        docs = last_message.content[results_start:]
    print("Documents found:", docs)

    prompt = (
        f"Based on these research documents, summarize the latest advancements in AI:\n"
        f"Question: {question}\n"
        f"Documents: {docs}\n"
        "Focus on extracting and synthesizing the key findings from the research papers."
    )
    response_text = llm(prompt)
    print("Final Answer:", response_text)
    return {"messages": [AIMessage(content=response_text)]}

def rewrite(state: AgentState):
    print("---REWRITE QUESTION---")
    messages = state["messages"]
    original_question = messages[0].content if len(messages) > 0 else "N/A"
    prompt = f"Rewrite this question to be more specific and clearer: {original_question}"
    response_text = llm(prompt)
    print("Rewritten question:", response_text)
    return {"messages": [AIMessage(content=response_text)]}

def custom_tools_condition(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content

    print("Checking tools condition:", content)
    # This pattern check remains from your original workflow. Define tools_pattern accordingly.
    if tools_pattern.match(content):
        print("Moving to retrieve...")
        return "tools"
    print("Moving to END...")
    return END

workflow = StateGraph(AgentState)

# Define workflow nodes
workflow.add_node("agent", agent)
retrieve_node = ToolNode(tools)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

workflow.add_conditional_edges("retrieve", simple_grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the workflow
app = workflow.compile()

def process_question(user_question, app, config):
    """Process user question through the workflow"""
    events = []
    for event in app.stream({"messages": [("user", user_question)]}, config):
        events.append(event)
    return events

def main():
    st.set_page_config(
        page_title="AI Research & Development Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin-top: 20px;
    }
    .data-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .research-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .dev-box {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with data display
    with st.sidebar:
        st.header("📚 Available Data")
        st.subheader("Research Database")
        for text in research_texts:
            st.markdown(f'<div class="data-box research-box">{text}</div>', unsafe_allow_html=True)
        st.subheader("Development Database")
        for text in development_texts:
            st.markdown(f'<div class="data-box dev-box">{text}</div>', unsafe_allow_html=True)

    st.title("🤖 AI Research & Development Assistant")
    st.markdown("---")

    # Query Input
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the latest advancement in AI research?")

    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("🔍 Get Answer", use_container_width=True):
            if query:
                with st.spinner('Processing your question...'):
                    events = process_question(query, app, {"configurable": {"thread_id": "1"}})
                    for event in events:
                        if 'agent' in event:
                            with st.expander("🔄 Processing Step", expanded=True):
                                content = event['agent']['messages'][0].content
                                if "Results:" in content:
                                    st.markdown("### 📑 Retrieved Documents:")
                                    docs_start = content.find("Results:")
                                    docs = content[docs_start:]
                                    st.info(docs)
                        elif 'generate' in event:
                            st.markdown("### ✨ Final Answer:")
                            st.success(event['generate']['messages'][0].content)
            else:
                st.warning("⚠️ Please enter a question first!")
    with col2:
        st.markdown("""
        ### 🎯 How to Use
        1. Type your question in the text box
        2. Click "Get Answer" to process
        3. View retrieved documents and final answer
        
        ### 💡 Example Questions
        - What are the latest advancements in AI research?
        - What is the status of Project A?
        - What are the current trends in machine learning?
        """)

if __name__ == "__main__":
    main()
