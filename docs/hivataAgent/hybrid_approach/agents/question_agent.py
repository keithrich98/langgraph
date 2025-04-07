# question_agent.py
# No need to import task decorator
from typing import Dict, List, Any
from hivataAgent.hybrid_approach.core.state import (
    SessionState, 
    get_formatted_current_question, 
    add_to_extraction_queue, 
    create_updated_state
)

# Import logger
from hivataAgent.hybrid_approach.config.logging_config import logger

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

def initialize_questions(state: SessionState) -> SessionState:
    """Initialize the questionnaire with questions."""
    logger.info("Initializing questionnaire state")
    
    # Load questions
    questions = get_questions()
    
    # Create a temporary state with questions to get the formatted first question
    temp_state = state.update(questions=questions, current_index=0)
    conversation_history = [{"role": "system", "content": get_formatted_current_question(temp_state) or "Welcome to the questionnaire."}]
    
    # Create a new state object with updated values (using Pydantic update pattern)
    updated_state = state.update(
        questions=questions,
        current_index=0,
        is_complete=False,
        responses={},
        verified_responses={},
        verification_messages={},
        verified_answers={},
        extracted_terms={},
        term_extraction_queue=[],
        conversation_history=conversation_history
    )
    
    logger.debug(f"State initialized with {len(updated_state.questions)} questions, starting at index {updated_state.current_index}")
    
    return updated_state

def present_next_question(state: SessionState) -> SessionState:
    """Present the next question in the sequence."""
    logger.info(f"Presenting next question (index: {state.current_index})")
    
    # Check if we've completed all questions
    if state.current_index >= len(state.questions):
        logger.info("All questions have been completed, marking questionnaire as complete")
        
        new_conversation_history = state.conversation_history.copy() + [{
            "role": "system", 
            "content": "Thank you for completing all the questions. Your responses have been recorded."
        }]
        
        return state.update(
            is_complete=True,
            conversation_history=new_conversation_history
        )
    
    # Format and add the current question
    formatted_question = get_formatted_current_question(state)
    if formatted_question:
        logger.debug(f"Adding question {state.current_index} to conversation history")
        new_conversation_history = state.conversation_history.copy() + [
            {"role": "system", "content": formatted_question}
        ]
        return state.update(conversation_history=new_conversation_history)
    else:
        logger.warning(f"Failed to get formatted question for index {state.current_index}")
        return state

def process_answer(state: SessionState, answer: str) -> SessionState:
    """Process a user's answer to the current question."""
    logger.info(f"Processing answer for question index {state.current_index}")
    logger.debug(f"Answer text (truncated): {answer[:50]}...")
    
    # Add the user's answer to conversation history
    new_conversation_history = state.conversation_history.copy() + [
        {"role": "user", "content": answer}
    ]
    
    # Store the answer for the current question
    current_index = state.current_index
    new_responses = state.responses.copy()
    new_responses[current_index] = answer
    
    logger.debug(f"Stored answer for question {current_index}, total responses: {len(new_responses)}")
    
    return state.update(
        conversation_history=new_conversation_history,
        responses=new_responses
    )

def advance_question(state: SessionState) -> SessionState:
    """Advance to the next question after verification."""
    current_index = state.current_index
    logger.info(f"Advancing from question {current_index} to next question")
    
    updated_state = state
    
    # If verification succeeded, add to the term extraction queue
    if state.verification_result.get("is_valid", False):
        question_text = state.questions[current_index]["text"]
        answer = state.responses.get(current_index, "")
        verification_message = state.verification_messages.get(current_index, "")
        
        # Create new verified_answers dictionary
        new_verified_answers = state.verified_answers.copy()
        new_verified_answers[current_index] = {
            "question": question_text,
            "answer": answer,
            "verification": verification_message
        }
        
        # Update state with new verified answers
        updated_state = updated_state.update(
            verified_answers=new_verified_answers
        )
        
        # Add to the term extraction queue
        updated_state = add_to_extraction_queue(updated_state, current_index)
        logger.info(f"Added question {current_index} to term extraction queue")
    
    # Update the current index
    final_state = updated_state.update(
        current_index=current_index + 1
    )
    
    if final_state.current_index >= len(final_state.questions):
        logger.info("Advanced to end of questionnaire, next step will mark as complete")
    else:
        logger.debug(f"Advanced to question index {final_state.current_index}")
    
    return final_state