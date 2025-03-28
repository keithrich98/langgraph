# question_answer.py - Refactored as a task

import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.func import task
from langchain_core.runnables import RunnableConfig

# Import shared state and helper
from state import ChatState, add_to_extraction_queue
from shared_memory import shared_memory  # not used here as entrypoint, but kept for consistency

# Set up the LLM for verification
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0
)

# Verification result model
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

def verify_answer(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> (bool, str):
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    formatted_history = ""
    for msg in conversation_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria."
        "Evaluate if the user's answer meets the specified requirements."
    )
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S ANSWER: {answer}\n\n"
        f"Evaluate if this answer meets all the requirements. Explain why if valid or specify missing items."
    )
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    try:
        response = llm_with_tool.invoke([system_message, user_message], config=config)
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            return result.is_valid, result.verification_message
        if hasattr(response, "content") and response.content:
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return is_valid, response.content
        return False, "Unable to verify your answer. Please provide more details."
    except Exception as e:
        print(f"Error in verification: {str(e)}")
        return False, f"There was an error verifying your answer: {str(e)}"

# def verify_answer(question: Dict[str, Any], answer: str, conversation_history: List[Dict[str, str]], config: Optional[RunnableConfig] = None) -> (bool, str):
#     """
#     Hard-coded verification function for testing purposes.
#     Always returns that the answer is valid.
    
#     Returns:
#         (True, "Your answer is valid. (Hardcoded verification)")
#     """
#     return True, "Your answer is valid. (Hardcoded verification)"

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
                "symptoms": "List each symptom experienced",
                "severity": "Include a severity rating (mild, moderate, severe)",
                "context": "Provide additional context about how symptoms impact daily life"
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

def debug_state(prefix: str, state: ChatState):
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")

@task
def question_answer_task(
    action: Dict = None,
    state: ChatState = None,
    config: Optional[RunnableConfig] = None
) -> ChatState:
    """
    Task to process question-answer logic.
    Expects an action dict with "action": "start" or "answer" (with "answer" field).
    Updates the state accordingly.
    """
    # Initialize state if not provided; if state exists but has no questions, initialize them.
    if state is None:
        state = ChatState(questions=get_questions())
    elif not state.questions:
        state.questions = get_questions()

    debug_state("QA-Initial", state)
    
    if action is None:
        return state
    
    if action.get("action") == "start":
        state.current_question_index = 0
        state.is_complete = False
        state.responses = {}
        state.verified_answers = {}
        state.term_extraction_queue = []
        state.extracted_terms = {}
        
        question = state.questions[0]
        formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in question["requirements"].items()])
        prompt = f"{question['text']}\n\nRequirements:\n{formatted_requirements}"
        state.conversation_history = [{"role": "ai", "content": prompt}]
        
        debug_state("QA-AfterStart", state)
    
    elif action.get("action") == "answer" and not state.is_complete:
        answer = action.get("answer", "")
        state.conversation_history.append({"role": "human", "content": answer})
        debug_state("QA-AfterHumanAnswer", state)
        
        current_question = state.questions[state.current_question_index]
        is_valid, verification_message = verify_answer(current_question, answer, state.conversation_history, config)
        state.conversation_history.append({"role": "ai", "content": verification_message})
        debug_state("QA-AfterVerification", state)
        
        if is_valid:
            state.responses[state.current_question_index] = answer
            state.verified_answers[state.current_question_index] = {
                "question": current_question["text"],
                "answer": answer,
                "verification": verification_message
            }
            state = add_to_extraction_queue(state, state.current_question_index)
            state.current_question_index += 1
            if state.current_question_index >= len(state.questions):
                state.is_complete = True
                state.conversation_history.append({
                    "role": "ai", 
                    "content": "Thank you for completing all the questions. Your responses have been recorded."
                })
            else:
                next_question = state.questions[state.current_question_index]
                formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in next_question["requirements"].items()])
                prompt = f"{next_question['text']}\n\nRequirements:\n{formatted_requirements}"
                state.conversation_history.append({"role": "ai", "content": prompt})
            debug_state("QA-AfterValidAnswer", state)
        else:
            debug_state("QA-AfterInvalidAnswer", state)
    
    debug_state("QA-Final", state)
    return state