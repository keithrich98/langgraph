# question_agent.py
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
from state import ChatState, get_state_for_thread, update_state_for_thread
from logging_config import logger

logger.info("Initializing question_agent.py")

def get_questions():
    """Return the list of questions for the questionnaire."""
    logger.info("Getting question list")
    questions = [
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
    logger.debug(f"Returning {len(questions)} questions")
    return questions

@tool
def ask_question_tool(thread_id: str) -> str:
    """
    Get the current question to ask the patient.
    
    Args:
        thread_id: The ID of the current conversation thread
        
    Returns:
        A formatted question with requirements
    """
    logger.info(f"ask_question_tool called for thread_id: {thread_id}")
    
    # Get or initialize state
    state = get_state_for_thread(thread_id)
    if not state:
        logger.info(f"No state found for thread_id: {thread_id}, initializing new state")
        # Initialize new state
        state = ChatState(
            questions=get_questions(),
            current_question_index=0,
            is_complete=False,
            conversation_history=[]
        )
        update_state_for_thread(thread_id, state)
    
    logger.debug(f"State retrieved - current question index: {state.current_question_index}")
    
    # Check if questionnaire is complete
    if state.is_complete:
        logger.info("Questionnaire is already complete")
        return "The questionnaire has been completed. Thank you for your participation."
    
    # Get current question
    if state.current_question_index >= len(state.questions):
        logger.info(f"Reached end of questions, marking as complete")
        state.is_complete = True
        update_state_for_thread(thread_id, state)
        return "The questionnaire has been completed. Thank you for your participation."
    
    current_question = state.questions[state.current_question_index]
    logger.debug(f"Current question: {current_question['text']}")
    
    formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in current_question["requirements"].items()])
    
    # Format the question with requirements
    formatted_question = f"{current_question['text']}\n\nRequirements:\n{formatted_requirements}"
    logger.debug(f"Formatted question: {formatted_question}")
    
    return formatted_question

@tool
def advance_to_next_question(thread_id: str, answer: str) -> str:
    """
    Advance to the next question in the questionnaire.
    
    Args:
        thread_id: The ID of the current conversation thread
        answer: The verified answer to the current question
        
    Returns:
        Confirmation message and the next question
    """
    logger.info(f"advance_to_next_question called for thread_id: {thread_id}")
    
    # Get state
    state = get_state_for_thread(thread_id)
    if not state:
        logger.error(f"No state found for thread_id: {thread_id}")
        return "Error: No active questionnaire session found."
    
    # Store the answer for the current question
    current_index = state.current_question_index
    logger.debug(f"Storing answer for question index: {current_index}")
    state.responses[current_index] = answer
    
    # Advance to the next question
    state.current_question_index += 1
    logger.info(f"Advanced to question index: {state.current_question_index}")
    
    # Check if we've completed all questions
    if state.current_question_index >= len(state.questions):
        logger.info("Reached end of questions, marking as complete")
        state.is_complete = True
        update_state_for_thread(thread_id, state)
        return "All questions have been completed. Thank you for participating in this questionnaire."
    
    # Update state
    update_state_for_thread(thread_id, state)
    
    # Return confirmation and next question
    next_question = ask_question_tool(thread_id)
    logger.debug(f"Next question: {next_question}")
    return f"Answer recorded. Moving to the next question:\n\n{next_question}"