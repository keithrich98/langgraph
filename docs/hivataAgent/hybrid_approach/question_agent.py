# question_agent.py
from langgraph.func import task
from typing import Dict, List, Any
from state import SessionState, get_formatted_current_question

# Import logger
from logging_config import logger

def get_questions():
    """
    Returns a list of questions with their requirements.
    """
    logger.debug("Loading questions from get_questions function")
    questions = [
        {
            "text": "At what age were you diagnosed with polymicrogyria, and what were the primary signs or symptoms?",
            "requirements": {
                "age": "Provide age",
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
    logger.debug(f"Loaded {len(questions)} questions")
    return questions

@task
def initialize_questions(state: SessionState) -> SessionState:
    """Initialize the questionnaire with questions."""
    logger.info("Initializing questionnaire state")
    
    # Load questions
    state.questions = get_questions()
    state.current_index = 0
    state.is_complete = False
    state.responses = {}
    state.verified_responses = {}
    state.verification_messages = {}
    state.extracted_terms = {}
    state.term_extraction_queue = []
    
    logger.debug(f"State initialized with {len(state.questions)} questions, starting at index {state.current_index}")
    
    # Initialize with first question
    formatted_question = get_formatted_current_question(state)
    if formatted_question:
        logger.debug("Adding first question to conversation history")
        state.conversation_history = [{"role": "system", "content": formatted_question}]
    else:
        logger.warning("Failed to get formatted question - no questions available")
    
    return state

@task
def present_next_question(state: SessionState) -> SessionState:
    """Present the next question in the sequence."""
    logger.info(f"Presenting next question (index: {state.current_index})")
    
    # Check if we've completed all questions
    if state.current_index >= len(state.questions):
        logger.info("All questions have been completed, marking questionnaire as complete")
        state.is_complete = True
        state.conversation_history.append({
            "role": "system", 
            "content": "Thank you for completing all the questions. Your responses have been recorded."
        })
        return state
    
    # Format and add the current question
    formatted_question = get_formatted_current_question(state)
    if formatted_question:
        logger.debug(f"Adding question {state.current_index} to conversation history")
        state.conversation_history.append({"role": "system", "content": formatted_question})
    else:
        logger.warning(f"Failed to get formatted question for index {state.current_index}")
    
    return state

@task
def process_answer(state: SessionState, answer: str) -> SessionState:
    """Process a user's answer to the current question."""
    logger.info(f"Processing answer for question index {state.current_index}")
    logger.debug(f"Answer text (truncated): {answer[:50]}...")
    
    # Add the user's answer to conversation history
    state.conversation_history.append({"role": "user", "content": answer})
    
    # Store the answer for the current question
    current_index = state.current_index
    state.responses[current_index] = answer
    logger.debug(f"Stored answer for question {current_index}, total responses: {len(state.responses)}")
    
    return state

@task
def advance_question(state: SessionState) -> SessionState:
    """Advance to the next question after verification."""
    current_index = state.current_index
    logger.info(f"Advancing from question {current_index} to next question")
    
    state.current_index += 1
    
    if state.current_index >= len(state.questions):
        logger.info("Advanced to end of questionnaire, next step will mark as complete")
    else:
        logger.debug(f"Advanced to question index {state.current_index}")
    
    return state