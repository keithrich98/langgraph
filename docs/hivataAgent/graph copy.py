# graph.py - Updated to include LLM verification with short-term memory

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import MemorySaver

# Set up checkpointer
memory = MemorySaver()

# Set up the LLM for verification
from langchain_openai import ChatOpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY must be set in your environment.")
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

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

def verify_answer(question: Dict[str, Any], answer: str, conversation_history: List[Dict[str, str]]) -> (bool, str):
    """
    Uses the LLM to verify if the user's answer meets the requirements.
    The full conversation history is included so the LLM can see all prior context.
    If the answer is valid, the LLM should respond with a message that includes "Valid".
    Otherwise, it will ask follow-up questions.
    """
    # Build a prompt with full context
    prompt = (
        "You are an expert validator for a medical questionnaire. "
        "Evaluate if the user's answer meets the following requirements.\n\n"
        f"Question: {question['text']}\n"
        f"Requirements: {question['requirements']}\n\n"
        "Conversation so far:\n"
    )
    for msg in conversation_history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"\nUser's Answer: {answer}\n\n"
    prompt += (
        "If the answer meets the requirements, reply with a short message that includes the word 'Valid'. "
        "If not, provide clear follow-up questions asking for clarification or more detail."
    )
    # Invoke the LLM with the prompt
    response = llm.invoke(prompt)
    verification_message = response  # assuming response is a plain text string
    is_valid = "Valid" in verification_message
    return is_valid, verification_message

# The workflow - updated to include verification step
@entrypoint(checkpointer=memory)
def questionnaire_workflow(action: Dict = None, *, previous: ChatState = None) -> ChatState:
    """
    A questionnaire workflow with a verification step.
    
    Actions:
      {"action": "start"} - Start a new questionnaire session.
      {"action": "answer", "answer": "user answer"} - Submit an answer.
    
    The workflow uses an LLM to validate the answer against the question's requirements.
    If the answer is not valid, the LLM asks follow-up questions until the answer is acceptable.
    """
    # Use previous state or initialize a new one
    state = previous if previous is not None else ChatState(questions=get_questions())
    
    if action is None:
        # No action provided, just return current state
        return state
    
    if action.get("action") == "start":
        # Initialize the session
        state.current_question_index = 0
        state.is_complete = False
        state.responses = {}
        question = state.questions[0]
        prompt = f"{question['text']}\nRequirements: {question['requirements']}"
        state.conversation_history = [{"role": "ai", "content": prompt}]
    
    elif action.get("action") == "answer" and not state.is_complete:
        answer = action.get("answer", "")
        # Append the user's answer to the conversation
        state.conversation_history.append({"role": "human", "content": answer})
        
        # Perform verification using the LLM and the full conversation history (i.e., short-term memory)
        current_question = state.questions[state.current_question_index]
        is_valid, verification_message = verify_answer(current_question, answer, state.conversation_history)
        # Append the verification message from the LLM to the conversation
        state.conversation_history.append({"role": "ai", "content": verification_message})
        
        if is_valid:
            # Valid answer: store and move to the next question
            state.responses[state.current_question_index] = answer
            state.current_question_index += 1
            if state.current_question_index >= len(state.questions):
                state.is_complete = True
                state.conversation_history.append({
                    "role": "ai", 
                    "content": "Thank you for completing all the questions. Your responses have been recorded."
                })
            else:
                # Present the next question
                next_question = state.questions[state.current_question_index]
                prompt = f"{next_question['text']}\nRequirements: {next_question['requirements']}"
                state.conversation_history.append({"role": "ai", "content": prompt})
        else:
            # If the answer is invalid, the follow-up questions remain visible.
            # The same question remains active so the user can try answering again.
            pass
    
    return state
