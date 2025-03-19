# graph.py - Simplified approach withouth HIL API

from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, List, Any
from dataclasses import dataclass, field

# Checkpointer
memory = MemorySaver()

# State definition
@dataclass
class ChatState:
    current_question_index: int = 0
    questions: List[Dict] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    responses: Dict[int, str] = field(default_factory=dict)
    is_complete: bool = False

# Questions list
def get_questions():
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

# The workflow - without any task or interrupt
@entrypoint(checkpointer=memory)
def questionnaire_workflow(action: Dict = None, *, previous: ChatState = None) -> ChatState:
    """
    A simplified questionnaire workflow that uses direct actions instead of interrupts.
    
    Args:
        action: A dictionary with the action to perform
              {"action": "start"} - Start a new questionnaire
              {"action": "answer", "answer": "user answer"} - Submit an answer
        previous: Previous state from checkpoint
    
    Returns:
        Updated ChatState
    """
    # Use previous state or initialize a new one
    state = previous if previous is not None else ChatState(questions=get_questions())
    
    # Process the action
    if action is None:
        # No action, just return current state
        return state
    
    if action.get("action") == "start":
        # Initialize or reset
        state.current_question_index = 0
        state.is_complete = False
        state.responses = {}
        
        # Add first question to conversation history
        question = state.questions[0]
        prompt = f"{question['text']}\nRequirements: {question['requirements']}"
        state.conversation_history = [{"role": "ai", "content": prompt}]
    
    elif action.get("action") == "answer" and not state.is_complete:
        # Get the answer
        answer = action.get("answer", "")
        
        # Add to conversation history and responses
        state.conversation_history.append({"role": "human", "content": answer})
        state.responses[state.current_question_index] = answer
        
        # Move to next question
        state.current_question_index += 1
        
        # Check if we've completed all questions
        if state.current_question_index >= len(state.questions):
            state.is_complete = True
            state.conversation_history.append({
                "role": "ai", 
                "content": "Thank you for completing all the questions. Your responses have been recorded."
            })
        else:
            # Add next question to conversation history
            question = state.questions[state.current_question_index]
            prompt = f"{question['text']}\nRequirements: {question['requirements']}"
            state.conversation_history.append({"role": "ai", "content": prompt})
    
    return state