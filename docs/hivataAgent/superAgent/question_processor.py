# question_processor.py - Task for question handling

from typing import Dict, List, Any, Optional
from langgraph.func import task
from langchain_core.runnables import RunnableConfig

# Import shared state and helpers
from state import ChatState, add_to_extraction_queue
# Import the logger from logging_config
from logging_config import logger

def get_questions():
    """
    Returns a list of questions with their requirements for the polymicrogyria questionnaire.
    """
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

@task
def question_processor_task(
    action: Dict = None,
    state: ChatState = None,
    config: Optional[RunnableConfig] = None
) -> ChatState:
    """
    Task to process question handling logic.
    Initializes questionnaire or processes valid answers to move to the next question.
    Does not handle verification - that's done by the verification_task.
    
    Actions:
    - "start": Initialize the questionnaire and send the first question
    - "verified_answer": Process a verified answer and move to the next question if valid
    """
    # Initialize state if not provided; if state exists but has no questions, initialize them.
    if state is None:
        state = ChatState(questions=get_questions())
    elif not state.questions:
        state.questions = get_questions()

    # Log key details of initial state
    logger.debug(f"QP-Initial: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"QP-Initial: current_question_index: {state.current_question_index}")
    logger.debug(f"QP-Initial: responses count: {len(state.responses)}")
    logger.debug(f"QP-Initial: thread_id: {id(state)}")
    
    if action is None:
        return state
    
    # Process start action - initialize the questionnaire
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
        
        # Log key details after start
        logger.debug(f"QP-AfterStart: conversation_history length: {len(state.conversation_history)}")
        logger.debug(f"QP-AfterStart: current_question_index: {state.current_question_index}")
        logger.debug(f"QP-AfterStart: responses count: {len(state.responses)}")
        logger.debug(f"QP-AfterStart: thread_id: {id(state)}")
    
    # Process valid answer - update state and move to next question
    elif action.get("action") == "verified_answer" and not state.is_complete:
        # Expecting the verification_task to have already verified the answer
        # and appended both user answer and verification response to conversation history
        
        # Get the data from the action
        question_index = action.get("question_index", state.current_question_index)
        answer = action.get("answer", "")
        verification_message = action.get("verification_message", "")
        is_valid = action.get("is_valid", False)
        
        # Log key details after receiving verification results
        logger.debug(f"QP-AfterVerification: question_index: {question_index}, is_valid: {is_valid}")
        logger.debug(f"QP-AfterVerification: conversation_history length: {len(state.conversation_history)}")
        
        if is_valid:
            # Store the verified answer and prepare for term extraction
            state.responses[question_index] = answer
            state.verified_answers[question_index] = {
                "question": state.questions[question_index]["text"],
                "answer": answer,
                "verification": verification_message
            }
            state = add_to_extraction_queue(state, question_index)
            
            # Move to the next question
            state.current_question_index += 1
            if state.current_question_index >= len(state.questions):
                state.is_complete = True
                state.conversation_history.append({
                    "role": "ai", 
                    "content": "Thank you for completing all the questions. Your responses have been recorded."
                })
            else:
                # Prepare the next question
                next_question = state.questions[state.current_question_index]
                formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in next_question["requirements"].items()])
                prompt = f"{next_question['text']}\n\nRequirements:\n{formatted_requirements}"
                state.conversation_history.append({"role": "ai", "content": prompt})
            
            # Log key details after valid answer
            logger.debug(f"QP-AfterValidAnswer: conversation_history length: {len(state.conversation_history)}")
            logger.debug(f"QP-AfterValidAnswer: current_question_index: {state.current_question_index}")
            logger.debug(f"QP-AfterValidAnswer: responses count: {len(state.responses)}")
    
    # Log key details of final state
    logger.debug(f"QP-Final: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"QP-Final: current_question_index: {state.current_question_index}")
    logger.debug(f"QP-Final: responses count: {len(state.responses)}")
    
    return state