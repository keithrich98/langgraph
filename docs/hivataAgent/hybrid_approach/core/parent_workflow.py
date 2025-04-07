# parent_workflow.py
import uuid
import threading
import time
from typing import Dict, Any, Optional, List, Literal, cast
from langgraph.func import entrypoint
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

def schedule_extraction_task(thread_id: str) -> None:
    """Schedule term extraction to be processed asynchronously."""
    logger.debug(f"Scheduling term extraction for thread_id: {thread_id}")
    # Note: Uses the trigger_extraction_in_thread function to avoid circular references
    trigger_extraction_in_thread(thread_id)
    return None

# These helper functions have been removed as we now call agent functions directly

# Function to determine next step based on state
def decide_next_step(state: SessionState) -> str:
    """Determine the next step based on current state."""
    # Check action type first
    action = state.current_action or {}
    action_type = action.get("action", "")
    
    if action_type == "start":
        return "initialize"
    
    elif action_type == "answer":
        if state.is_complete:
            return "complete"
        else:
            return "process_answer"
    
    elif action_type == "extract_terms":
        return "extract_terms"
        
    return "invalid_action"

# Main workflow function using functional API with improved structure
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
    state = state.update(current_action=action)
    
    try:
        # Determine next step based on action and state
        next_step = decide_next_step(state)
        logger.debug(f"Next step determined: {next_step}")
        
        # Execute appropriate action based on the next step
        if next_step == "initialize":
            # Initialize the questionnaire
            logger.info(f"Starting new questionnaire for thread_id: {thread_id}")
            return initialize_questions(state)
            
        elif next_step == "process_answer":
            # Get the answer from the action
            answer = action.get("answer", "")
            
            # Process the answer
            processed_state = process_answer(state, answer)
            
            # Verify the answer
            verified_state = verify_answer(processed_state)
            
            # Check verification result and determine next steps
            if verified_state.verification_result.get("is_valid", False):
                # Answer is valid, advance to next question
                advanced_state = advance_question(verified_state)
                
                # Determine if we need to present the next question
                if not advanced_state.is_complete and advanced_state.current_index < len(advanced_state.questions):
                    # Present the next question
                    updated_state = present_next_question(advanced_state)
                else:
                    # Mark as complete if we've reached the end
                    updated_state = advanced_state.update(is_complete=True)
                
                # If there are items queued for term extraction, trigger it asynchronously
                if updated_state.term_extraction_queue:
                    logger.debug(f"Triggering async term extraction. Queue: {updated_state.term_extraction_queue}")
                    schedule_extraction_task(thread_id)
                
                return updated_state
            else:
                # Answer is invalid, return the state with verification feedback
                logger.debug(f"Answer is invalid, requesting correction")
                return verified_state
                
        elif next_step == "extract_terms":
            # Process term extraction
            logger.info(f"Processing term extraction for thread_id: {thread_id}")
            extracted_state = term_extraction_task(state, {"action": "extract_terms"})
            
            # If there are more items, trigger another extraction
            if extracted_state.term_extraction_queue:
                logger.debug("More items in queue, triggering next extraction")
                schedule_extraction_task(thread_id)
                
            return extracted_state
            
        else:
            # Invalid action or state
            logger.warning(f"Invalid action or questionnaire already complete: {action}")
            return state
    
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}", exc_info=True)
        # Store error in state for API responses
        state = state.update(error=str(e))
    
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
        # Using entrypoint.final to separate API response from stored state
        result = workflow.invoke(
            {"action": "start"}, 
            config={"configurable": {"thread_id": thread_id}}
        )
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
    def _run_extraction():
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
    
    # Create and start a new thread
    extraction_thread = threading.Thread(target=_run_extraction)
    extraction_thread.daemon = True
    extraction_thread.start()
    return None

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
        
        # Always use the values property for the state
        state_values = state.values if hasattr(state, 'values') else state
        
        return {
            "session_id": session_id,
            **to_dict(state_values)
        }
    except Exception as e:
        logger.error(f"Error retrieving session state: {str(e)}", exc_info=True)
        return None