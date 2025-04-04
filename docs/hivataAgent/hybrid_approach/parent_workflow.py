# parent_workflow.py
import uuid
from typing import Dict, Any, Optional
from langgraph.func import entrypoint
from langchain_core.runnables import RunnableConfig

# Import shared memory and state management
from shared_memory import shared_memory
from state import SessionState, to_dict

# Import agent tasks
from question_agent import (
    initialize_questions, 
    present_next_question, 
    process_answer, 
    advance_question
)
from verification_agent import verify_answer

# Import logger
from logging_config import logger

@entrypoint(checkpointer=shared_memory)
def workflow(action: Dict[str, Any] = None, *, previous: SessionState = None, config: Optional[RunnableConfig] = None) -> SessionState:
    """
    Main workflow orchestrating the question and verification flow.
    
    Actions:
    - {"action": "start"}: Initialize the questionnaire
    - {"action": "answer", "answer": "..."}: Process a user answer
    """
    # Get thread_id for logging context
    thread_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
    logger.debug(f"Workflow called with thread_id: {thread_id}, action: {action}")
    
    # Initialize or use previous state with detailed logging
    if previous:
        logger.debug(f"Using previous state: current_index={previous.current_index}, " +
                    f"responses={len(previous.responses)}, " +
                    f"conversation_history={len(previous.conversation_history)}, " +
                    f"verified_responses={len(previous.verified_responses)}")
        state = previous
    else:
        logger.warning(f"No previous state found for thread_id: {thread_id}, creating new SessionState")
        state = SessionState()
    
    if not action:
        logger.debug(f"No action provided, returning current state for thread_id: {thread_id}")
        return state
    
    # Start the questionnaire
    if action.get("action") == "start":
        logger.info(f"Starting new questionnaire for thread_id: {thread_id}")
        
        # Initialize questions and present first question
        state = initialize_questions(state).result()
        logger.debug(f"Questionnaire initialized with {len(state.questions)} questions")
        
        return state
    
    # Process an answer
    elif action.get("action") == "answer" and not state.is_complete:
        answer = action.get("answer", "")
        logger.info(f"Processing answer for question {state.current_index} (thread_id: {thread_id})")
        logger.debug(f"Answer text: {answer[:50]}...")  # Log first 50 chars of answer
        logger.debug(f"Current responses before processing: {state.responses}")
        
        # 1. Add the answer to state
        state = process_answer(state, answer).result()
        logger.debug(f"Answer processed and added to state for thread_id: {thread_id}")
        logger.debug(f"Current responses after processing: {state.responses}")
        logger.debug(f"Conversation history length: {len(state.conversation_history)}")
        
        # 2. Verify the answer
        logger.debug(f"Starting verification for question {state.current_index}")
        state = verify_answer(state).result()
        
        # Log verification result
        verification_result = state.verification_result
        is_valid = verification_result.get("is_valid", False)
        logger.info(f"Verification result for question {state.current_index}: valid={is_valid}")
        logger.debug(f"Verified responses after verification: {state.verified_responses}")
        
        # 3. If the answer is valid, advance to the next question
        if is_valid:
            logger.debug(f"Answer is valid, advancing to next question for thread_id: {thread_id}")
            state = advance_question(state).result()
            logger.debug(f"Advanced to question index {state.current_index}")
            
            if state.current_index >= len(state.questions):
                logger.info(f"Questionnaire completed for thread_id: {thread_id}")
                state.is_complete = True
                logger.debug(f"Questionnaire marked as complete: is_complete={state.is_complete}")
            else:
                logger.debug(f"Presenting question {state.current_index} for thread_id: {thread_id}")
                state = present_next_question(state).result()
                logger.debug(f"Conversation history after presenting next question: {len(state.conversation_history)} messages")
        else:
            logger.debug(f"Answer is invalid, requesting correction for thread_id: {thread_id}")
            logger.debug(f"Conversation history after verification failure: {len(state.conversation_history)} messages")
    else:
        logger.warning(f"Invalid action or questionnaire already complete: {action} (thread_id: {thread_id})")
        if state.is_complete:
            logger.debug("Questionnaire is already marked as complete")
        
    # Final state check before returning
    logger.debug(f"Final state before return: current_index={state.current_index}, " +
                f"responses_count={len(state.responses)}, " +
                f"conversation_history_count={len(state.conversation_history)}, " +
                f"verified_responses_count={len(state.verified_responses)}")
    
    return state

# Helper functions for API integration
def start_session():
    """Start a new questionnaire session."""
    thread_id = str(uuid.uuid4())
    logger.info(f"Creating new session with thread_id: {thread_id}")
    
    try:
        result = workflow.invoke({"action": "start"}, config={"configurable": {"thread_id": thread_id}})
        logger.debug(f"Session started successfully: {thread_id}")
        
        return {
            "session_id": thread_id,
            **to_dict(result)
        }
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}", exc_info=True)
        raise

def process_user_answer(session_id: str, answer: str):
    """Process a user's answer and get the next question or verification feedback."""
    logger.info(f"Processing user answer for session: {session_id}")
    
    try:
        # Make sure thread_id is being passed correctly
        result = workflow.invoke(
            {"action": "answer", "answer": answer}, 
            config={"configurable": {"thread_id": session_id}}
        )
        
        logger.debug(f"Answer processed for session {session_id}, is_complete: {result.is_complete}")
        
        return {
            "session_id": session_id,
            **to_dict(result)
        }
    except Exception as e:
        logger.error(f"Error processing answer for session {session_id}: {str(e)}", exc_info=True)
        raise

def get_session_state(session_id: str):
    """Get the current state of a session."""
    logger.info(f"Retrieving session state: {session_id}")
    
    try:
        # Make sure you're passing the thread_id correctly
        state = workflow.get_state(config={"configurable": {"thread_id": session_id}})
        if not state:
            logger.warning(f"No state found for session: {session_id}")
            return None
        
        # Make sure we're using the actual state values, not creating new ones
        logger.debug(f"Successfully retrieved state for session {session_id}")
        
        return {
            "session_id": session_id,
            **to_dict(state.values)  # Ensure we're using state.values here
        }
    except Exception as e:
        logger.error(f"Error retrieving session state: {str(e)}", exc_info=True)
        return None