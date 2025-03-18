# graph.py

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from typing import Literal, List, Dict, Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

# Import functional API components
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

@dataclass
class ChatState:
    current_question_index: int = 0
    questions: List[Dict[str, Dict[str, str]]] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    responses: Dict[int, str] = field(default_factory=dict)

model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

# Create a checkpointer for interrupt functionality
checkpointer = MemorySaver()

# Original StateGraph implementation (kept for backward compatibility)
def init_node(state: ChatState) -> ChatState:
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

# This is kept but renamed to distinguish it from the real interrupt function
def local_interrupt(prompt_obj) -> str:
    """Unused in your API code, for local testing only."""
    while True:
        answer = input(prompt_obj["prompt"] + "\nYour answer: ")
        if answer.strip() == "":
            print("Empty input detected. Please provide a valid answer.")
        else:
            return answer

def ask_node_api(state: ChatState) -> Command[Literal["increment_node"]]:
    """
    Sends the current question to the user (via the conversation_history).
    The user will see this question in the API response.
    """
    idx = state.current_question_index
    question_obj = state.questions[idx]
    prompt = f"{question_obj['text']}\nRequirements: {question_obj['requirements']}"
    state.conversation_history.append({"role": "ai", "content": prompt})
    return Command(goto="increment_node", update=state)

def increment_node(state: ChatState) -> Command[Literal["ask_node_api", END]]:
    """
    Increments the current question index.
    If all questions are done, we go to END.
    Otherwise, ask the next question.
    """
    state.current_question_index += 1
    if state.current_question_index >= len(state.questions):
        return Command(goto=END, update=state)
    return Command(goto="ask_node_api", update=state)

def validation_response(
    question_text: str,
    requirements: Dict[str, str],
    user_answer: str,
    conversation_so_far: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Calls GPT-4o to:
       1) Check if user_answer meets question requirements.
       2) If invalid, produce a follow-up question or clarification.
       3) If valid, indicate success.

    Returns a dict:
       {
         "valid": "true" or "false",
         "message": <some LLM text to show the user either confirming or clarifying>
       }
    """

    system_prompt = (
        "You are a validation assistant. You are given:\n"
        f"Question: {question_text}\n"
        f"Requirements: {requirements}\n"
        f"User Answer: {user_answer}\n\n"
        "Check if the user's answer meets the requirements. "
        "If it meets them, respond with a short JSON containing valid=true and a brief message. \n"
        "If not, respond with valid=false and a short follow-up or clarifying question. \n"
        "Format strictly in JSON with the keys: valid, message."
    )

    messages = []
    messages.append({"role": "system", "content": system_prompt})

    llm_response = model.invoke(messages)
    raw_text = llm_response.content.strip()

    import json
    try:
        parsed = json.loads(raw_text)
        if not all(k in parsed for k in ["valid", "message"]):
            return {"valid": "false", "message": raw_text}
        valid_value = str(parsed["valid"]).lower()
        return {
            "valid": valid_value,
            "message": parsed["message"]
        }
    except json.JSONDecodeError:
        return {
            "valid": "false",
            "message": raw_text
        }

# Build the original StateGraph
builder = StateGraph(ChatState)
builder.add_node("init_node", init_node)
builder.add_node("ask_node_api", ask_node_api)
builder.add_node("increment_node", increment_node)

builder.add_edge(START, "init_node")
builder.add_edge("init_node", "ask_node_api")
builder.add_edge("ask_node_api", "increment_node")
builder.add_edge("increment_node", END)

graph = builder.compile()

# ========== NEW FUNCTIONAL API IMPLEMENTATION ==========

# Helper function to initialize questions (same as in init_node)
def get_default_questions():
    return [
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

# Task-based implementations for the functional API
@task
def initialize_state():
    """Initialize the survey questions and state."""
    return {
        "questions": get_default_questions(),
        "current_index": 0,
        "conversation_history": [],
        "responses": {}
    }

@task
def present_question(state: Dict[str, Any]):
    """Present the current question to the user."""
    idx = state.get("current_index", 0)
    questions = state.get("questions", [])
    
    if idx >= len(questions):
        return {"finished": True}
    
    question = questions[idx]
    prompt = f"{question['text']}\nRequirements: {question['requirements']}"
    
    conversation = state.get("conversation_history", [])
    conversation.append({"role": "ai", "content": prompt})
    
    return {
        "conversation_history": conversation,
        "current_question": question,
        "current_index": idx
    }

@task
def collect_user_input(state: Dict[str, Any]):
    """Collect the user's answer using interrupt."""
    question = state.get("current_question")
    prompt = f"{question['text']}\nRequirements: {question['requirements']}"
    
    # Real interrupt implementation - this will pause the graph execution
    user_answer = interrupt({"prompt": prompt})
    
    conversation = state.get("conversation_history", [])
    conversation.append({"role": "human", "content": user_answer})
    
    return {
        "conversation_history": conversation,
        "user_answer": user_answer
    }

@task
def validate_user_answer(state: Dict[str, Any]):
    """Validate the user's answer against requirements."""
    question = state.get("current_question")
    user_answer = state.get("user_answer")
    conversation = state.get("conversation_history", [])
    
    validation_result = validation_response(
        question_text=question["text"],
        requirements=question["requirements"],
        user_answer=user_answer,
        conversation_so_far=conversation
    )
    
    conversation.append({"role": "ai", "content": validation_result["message"]})
    
    return {
        "conversation_history": conversation,
        "validation_result": validation_result
    }

@task
def process_validation_result(state: Dict[str, Any]):
    """Process validation result and determine next steps."""
    validation_result = state.get("validation_result", {})
    current_index = state.get("current_index", 0)
    user_answer = state.get("user_answer", "")
    
    if validation_result.get("valid") == "true":
        # Valid answer, store and move to next question
        responses = state.get("responses", {})
        responses[current_index] = user_answer
        
        return {
            "responses": responses,
            "current_index": current_index + 1,
            "proceed_to_next": True
        }
    else:
        # Invalid answer, need to try again
        return {
            "proceed_to_next": False
        }

# Main entrypoint that orchestrates the survey workflow
@entrypoint(checkpointer=checkpointer)
def survey_workflow(state=None):
    """Main survey workflow using the functional API."""
    # Initialize state if not provided
    if state is None:
        state = initialize_state().result()
    
    while not state.get("finished", False):
        # Present the current question
        state.update(present_question(state).result())
        
        if state.get("finished", False):
            break
        
        # Loop until we get a valid answer
        valid_answer = False
        while not valid_answer:
            # Collect user input
            state.update(collect_user_input(state).result())
            
            # Validate the user's answer
            state.update(validate_user_answer(state).result())
            
            # Process validation result
            state.update(process_validation_result(state).result())
            
            # Check if we can proceed
            valid_answer = state.get("proceed_to_next", False)
            if "proceed_to_next" in state:
                del state["proceed_to_next"]
    
    return state

# Export both the original and functional implementations
functional_graph = survey_workflow

if __name__ == "__main__":
    state = ChatState()
    state = init_node(state)
    final_state = graph.invoke(state)
    print("Final state:", final_state)