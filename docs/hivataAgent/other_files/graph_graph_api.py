# graph.py with HIL API

# --- Imports and Setup ---
from langgraph.graph import StateGraph, START, END  
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI           
from typing import Literal, List, Dict            
from dataclasses import dataclass, field         
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()  
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

# --- Define the ChatState Schema ---
@dataclass
class ChatState:
    current_question_index: int = 0
    questions: List[Dict[str, Dict[str, str]]] = field(default_factory=list)  # List of questions with requirements.
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # List to store conversation messages.
    responses: Dict[int, str] = field(default_factory=dict)                  # User responses keyed by question index.
    is_complete: bool = False  # Flag to indicate completion of all questions

# --- Instantiate the GPT-4 model (using gpt-4o) ---
model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

# --- Node 1: init_node ---
def init_node(state: ChatState) -> ChatState:
    """
    Initializes the state if 'questions' is empty.
    This node sets up the questions, conversation history, and responses.
    """
    if not state.questions:
        state.questions = [
            {
                "text": "At what age were you diagnosed with polymicrogyria, and what were the primary signs or symptoms?",
                "requirements": {
                    "age": "Provide age (based on birthdate)",
                    "diagnosis_date": "Provide the date of diagnosis",
                    "symptoms": "Describe the key signs and symptoms"
                }
            },
            {
                "text": "What symptoms or neurological issues do you experience, and how would you rate their severity?",
                "requirements": {
                    "symptoms": "List each symptom or neurological issue experienced",
                    "severity": "Include a severity rating for each (mild, moderate, severe)",
                    "context": "Provide any additional context about how symptoms impact daily life"
                }
            },
            {
                "text": "Can you describe the key findings from your brain imaging studies (MRI/CT)?",
                "requirements": {
                    "imaging_modality": "Specify the imaging modality used (MRI, CT, etc.)",
                    "findings": "Detail the main imaging findings",
                    "remarks": "Include any remarks from radiology reports"
                }
            }
        ]
        state.current_question_index = 0
        state.conversation_history = []
        state.responses = {}
        state.is_complete = False
    return state

# --- Node 2: ask_node ---
def ask_node(state: ChatState) -> Command[Literal["answer_node", "end_node"]]:
    """
    Adds the current question to the conversation history.
    This ensures the question is ready when the API returns the state.
    """
    idx = state.current_question_index

    # Make sure we don't go past the end of the questions
    if idx >= len(state.questions):
        # If we've already processed all questions, go to end_node
        return Command(goto="end_node", update=state)

    question_obj = state.questions[idx]
    prompt = f"{question_obj['text']}\nRequirements: {question_obj['requirements']}"
    state.conversation_history.append({"role": "ai", "content": prompt})

    return Command(goto="answer_node", update=state)

# --- Node 3: answer_node ---
def answer_node(state: ChatState) -> Command[Literal["increment_node", "end_node"]]:
    """
    Uses interrupt() to wait for the user's answer, then adds it to the conversation history.
    """
    idx = state.current_question_index

    # Make sure we have a question to answer
    if idx >= len(state.questions):
        return Command(goto="end_node", update=state)

    # Get answer using interrupt
    answer = interrupt({"prompt": "Please provide your answer:"})

    # Add the answer to the conversation history and responses
    state.conversation_history.append({"role": "human", "content": answer})
    state.responses[idx] = answer

    # Check if this was the last question
    if idx == len(state.questions) - 1:
        # If this is the last question, mark as complete and go to end_node
        state.is_complete = True
        return Command(goto="end_node", update=state)
    else:
        return Command(goto="increment_node", update=state)

# --- Node 4: increment_node ---
def increment_node(state: ChatState) -> Command[Literal["ask_node"]]:
    """
    Increments the question index.
    """
    state.current_question_index += 1
    return Command(goto="ask_node", update=state)

# --- Node 5: end_node ---
def end_node(state: ChatState) -> ChatState:
    """
    Explicitly marks the graph as complete.
    Performs any final processing needed.
    """
    state.is_complete = True
    # Add a completion message to the conversation history
    state.conversation_history.append({
        "role": "ai", 
        "content": "Thank you for completing all the questions. Your responses have been recorded."
    })
    return state

# --- Build the Graph ---
builder = StateGraph(ChatState)
builder.add_node("init_node", init_node)
builder.add_node("ask_node", ask_node)
builder.add_node("answer_node", answer_node)
builder.add_node("increment_node", increment_node)
builder.add_node("end_node", end_node)

# Set up edges
builder.add_edge(START, "init_node")
builder.add_edge("init_node", "ask_node")
# Note: answer_node and ask_node now return Commands, so we don't need these edges
# builder.add_edge("ask_node", "answer_node")
# builder.add_edge("answer_node", "increment_node")
# builder.add_edge("increment_node", "ask_node")
builder.add_edge("end_node", END)

# Create a checkpointer for persistence
memory = MemorySaver()

# Compile the graph with checkpointer
graph = builder.compile(checkpointer=memory)

# For testing with the CLI
if __name__ == "__main__":
    state = ChatState()
    state = init_node(state)
    final_state = graph.invoke(state, {"configurable": {"thread_id": "test"}})
    print("Final state:", final_state)