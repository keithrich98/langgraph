# parent_workflow.py
import uuid
import threading
import time
from typing import Dict, Any, Optional, List, Literal, cast
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

# Import shared memory and state management
from hivataAgent.hybrid_approach.core.shared_memory import shared_memory
from hivataAgent.hybrid_approach.core.state import (
    SessionState, 
    to_dict, 
    get_next_extraction_task, 
    create_updated_state
)

# Import agent tasks
from hivataAgent.hybrid_approach.agents.question_agent import (
    initialize_questions, 
    present_next_question, 
    process_answer, 
    advance_question
)
from hivataAgent.hybrid_approach.agents.verification_agent import verify_answer
from hivataAgent.hybrid_approach.agents.term_extractor_agent import term_extraction_task

# Import logger
from hivataAgent.hybrid_approach.config.logging_config import logger

# Define tasks for workflow
# These tasks are designed to be non-blocking and composable
# following LangGraph functional API best practices.
# They return futures directly instead of calling .result() internally,
# which allows for better orchestration in the main workflow.

@task
def schedule_extraction_task(thread_id: str) -> None:
    """Schedule term extraction to be processed asynchronously."""
    logger.debug(f"Scheduling term extraction for thread_id: {thread_id}")
    # Note: Uses the trigger_extraction_in_thread function to avoid circular references
    trigger_extraction_in_thread(thread_id)
    return None

# Removed @task decorator as initialize_questions no longer returns a Future
def start_questionnaire_task(state: SessionState) -> SessionState:
    """Initialize the questionnaire and present the first question."""
    logger.info("Starting new questionnaire")
    return initialize_questions(state)

# Removed @task decorator as process_answer no longer returns a Future
def process_answer_task(state: SessionState, answer: str) -> SessionState:
    """Process the user's answer and add it to state."""
    logger.info(f"Processing answer for question {state.current_index}")
    logger.debug(f"Answer text: {answer[:50]}...")  # Log first 50 chars of answer
    
    # Call the function directly
    return process_answer(state, answer)

# Removed @task decorator as verify_answer no longer returns a Future
def verify_answer_task(state: SessionState) -> SessionState:
    """Verify if the answer is valid according to requirements."""
    logger.debug(f"Starting verification for question {state.current_index}")
    
    # Call the function directly
    return verify_answer(state)

# Removed @task decorator as advance_question no longer returns a Future
def advance_question_task(state: SessionState) -> SessionState:
    """Advance to the next question if available, or mark as complete."""
    logger.debug(f"Answer is valid, advancing to next question")
    
    # Call the function directly
    return advance_question(state)

# Removed @task decorator as term_extraction_task no longer returns a Future
def extract_terms_task(state: SessionState) -> SessionState:
    """Extract medical terms from verified answers."""
    logger.info(f"Processing term extraction")
    
    if not state.term_extraction_queue:
        logger.debug("No items in term extraction queue")
        return state
    
    # Call the function directly
    return term_extraction_task(state, {"action": "extract_terms"})

# Main workflow function using functional API
@entrypoint(checkpointer=shared_memory)
def workflow(action: Dict[str, Any] = None, *, previous: SessionState = None, config: Optional[RunnableConfig] = None) -> SessionState:
    """
    Main workflow orchestrating the question and verification flow.
    
    Actions:
    - {"action": "start"}: Initialize the questionnaire
    - {"action": "answer", "answer": "..."}: Process a user answer
    - {"action": "extract_terms"}: Process the next term extraction task
    """
    # Get thread_id for logging context
    thread_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
    logger.debug(f"Workflow called with thread_id: {thread_id}, action: {action}")
    
    # Initialize or use previous state
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
    
    # Store the current action in state for reference
    state = create_updated_state(state, current_action=action)
    
    try:
        # Start the questionnaire
        if action.get("action") == "start":
            logger.info(f"Starting new questionnaire for thread_id: {thread_id}")
            # Execute task and ensure result is not a Future
            state = initialize_questions(state)
            # If initialize_questions returns a Future, we need to resolve it
            if hasattr(state, 'result'):
                state = state.result()
            return state
        
        # Process an answer
        elif action.get("action") == "answer" and not state.is_complete:
            answer = action.get("answer", "")
            
            # Process the answer and resolve any Future
            processed_state = process_answer(state, answer)
            if hasattr(processed_state, 'result'):
                processed_state = processed_state.result()
            
            # Verify the answer and resolve any Future
            verified_state = verify_answer(processed_state)
            if hasattr(verified_state, 'result'):
                verified_state = verified_state.result()
            state = verified_state
            
            # Check verification result and handle next steps
            is_valid = state.verification_result.get("is_valid", False)
            if is_valid:
                # Advance to next question and resolve any Future
                advanced_state = advance_question(state)
                if hasattr(advanced_state, 'result'):
                    advanced_state = advanced_state.result()
                state = advanced_state
                
                # Process presenting the next question if needed
                if not state.is_complete:
                    # Check if we've reached the end of the questionnaire
                    if state.current_index >= len(state.questions):
                        logger.info(f"Questionnaire completed")
                        state = create_updated_state(state, is_complete=True)
                    else:
                        # Present the next question and resolve any Future
                        logger.debug(f"Presenting question {state.current_index}")
                        next_state = present_next_question(state)
                        if hasattr(next_state, 'result'):
                            next_state = next_state.result()
                        state = next_state
                
                # If there are items queued for term extraction, trigger it asynchronously
                if state.term_extraction_queue:
                    logger.debug(f"Extraction: Triggering async term extraction. Queue: {state.term_extraction_queue}")
                    # Schedule term extraction asynchronously 
                    # Note: This doesn't affect state directly, so no need to resolve a future
                    schedule_extraction_task(thread_id)
            else:
                logger.debug(f"Answer is invalid, requesting correction for thread_id: {thread_id}")
        
        # Process term extraction
        elif action.get("action") == "extract_terms":
            logger.info(f"Processing term extraction for thread_id: {thread_id}")
            
            # Process extraction task and resolve any Future
            extracted_state = term_extraction_task(state, {"action": "extract_terms"})
            if hasattr(extracted_state, 'result'):
                extracted_state = extracted_state.result()
            state = extracted_state
            
            # If there are more items, trigger another extraction
            if state.term_extraction_queue:
                logger.debug("More items in queue, triggering next extraction")
                schedule_extraction_task(thread_id)
        
        else:
            logger.warning(f"Invalid action or questionnaire already complete: {action} (thread_id: {thread_id})")
    
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}", exc_info=True)
        # Store error in state for API responses
        state = create_updated_state(state, error=str(e))
    
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

def trigger_extraction_in_thread(thread_id: Optional[str] = None):
    """Background thread function to trigger term extraction."""
    try:
        logger.debug(f"Thread: Starting extraction thread for thread_id: {thread_id}")
        time.sleep(0.5)  # Short delay
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        
        # Invoke the workflow with extract_terms action
        workflow.invoke({"action": "extract_terms"}, config=config)
        logger.debug(f"Thread: Extraction completed for thread_id: {thread_id}")
    except Exception as e:
        import traceback
        logger.error(f"Thread: Error in extraction thread: {str(e)}")
        logger.error(traceback.format_exc())

def get_session_state(session_id: str):
    """Get the current state of a session."""
    logger.info(f"Retrieving session state: {session_id}")
    
    try:
        # Get state from the checkpointer
        state = workflow.get_state(config={"configurable": {"thread_id": session_id}})
        if not state:
            logger.warning(f"No state found for session: {session_id}")
            return None
        
        logger.debug(f"Successfully retrieved state for session {session_id}")
        
        return {
            "session_id": session_id,
            **to_dict(state.values)
        }
    except Exception as e:
        logger.error(f"Error retrieving session state: {str(e)}", exc_info=True)
        return None