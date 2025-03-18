# graph.py

# --- Imports and Setup ---
from langgraph.graph import StateGraph, START, END  # Used to build the graph.
from langgraph.types import Command                # Command objects combine state updates and control flow.
from langchain_openai import ChatOpenAI            # To instantiate a GPT-4 model.
from typing import Literal, List, Dict              # For type annotations.
from dataclasses import dataclass, field          # Dataclass for state schema with defaults.
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
    return state

# --- Override interrupt for testing with error handling ---
def interrupt(prompt_obj) -> str:
    """
    Interactive function: waits for user input from terminal.
    (Used only for interactive testing, not by the API.)
    """
    while True:
        answer = input(prompt_obj["prompt"] + "\nYour answer: ")
        if answer.strip() == "":
            print("Empty input detected. Please provide a valid answer.")
        else:
            return answer

# --- Node 2: ask_node_api ---
def ask_node_api(state: ChatState) -> Command[Literal["increment_node"]]:
    """
    Non-interactive version of ask_node for API usage.
    Instead of calling interrupt(), it simply appends the current question to the conversation history.
    This way, the API can return the question immediately without waiting for user input.
    """
    idx = state.current_question_index
    question_obj = state.questions[idx]
    prompt = f"{question_obj['text']}\nRequirements: {question_obj['requirements']}"
    state.conversation_history.append({"role": "ai", "content": prompt})
    return Command(goto="increment_node", update=state)

# --- Node 3: increment_node ---
def increment_node(state: ChatState) -> Command[Literal["ask_node_api", END]]:
    """
    Increments the current question index.
    If all questions have been answered, routes to END.
    Otherwise, loops back to ask_node_api for the next question.
    """
    state.current_question_index += 1
    if state.current_question_index >= len(state.questions):
        return Command(goto=END, update=state)
    return Command(goto="ask_node_api", update=state)

# --- Build the Graph ---
builder = StateGraph(ChatState)
builder.add_node("init_node", init_node)
builder.add_node("ask_node_api", ask_node_api)
builder.add_node("increment_node", increment_node)
builder.add_edge(START, "init_node")                # Graph starts at init_node.
builder.add_edge("init_node", "ask_node_api")         # After initialization, go to ask_node_api.
builder.add_edge("ask_node_api", "increment_node")    # After asking, move to increment_node.
builder.add_edge("increment_node", END)               # Terminate when done.
graph = builder.compile()

# Expose the graph object for import by the API.
if __name__ == "__main__":
    state = ChatState()
    state = init_node(state)
    final_state = graph.invoke(state)
    print("Final state:", final_state)
