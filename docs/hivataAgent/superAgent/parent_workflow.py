# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
import threading
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langchain_core.runnables import RunnableConfig

# Import the shared state and helper
from state import ChatState, get_next_extraction_task
# Import the refactored child tasks
from question_processor import question_processor_task
from answer_verifier import answer_verification_task
from term_extractor import term_extraction_task
# Use the shared memory checkpointer
from shared_memory import shared_memory
# Import the logger from logging_config
from logging_config import logger

@entrypoint(checkpointer=shared_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Parent workflow that coordinates all agent tasks.
    
    Supported actions:
      - {"action": "start"}: Initialize a new session.
      - {"action": "answer", "answer": "user answer"}: Process user answer.
      - {"action": "extract_terms"}: Process the next term extraction task.
      - {"action": "status"}: Report system status.
    """
    # Use previous state or initialize a new one.
    state = previous if previous is not None else ChatState()
    
    thread_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
    logger.debug(f"Running with thread_id: {thread_id}")
    
    # Log key details of initial state
    logger.debug(f"Parent-Initial: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"Parent-Initial: current_question_index: {state.current_question_index}")
    logger.debug(f"Parent-Initial: responses count: {len(state.responses)}")
    logger.debug(f"Parent-Initial: thread_id: {id(state)}")
    
    if action is None:
        return state

    # Process "start" action using the question processor directly
    if action.get("action") == "start":
        state = question_processor_task(action, state, config=config).result()
        
        # Log after processing the start action
        logger.debug(f"Parent-AfterStart: conversation_history length: {len(state.conversation_history)}")
        logger.debug(f"Parent-AfterStart: current_question_index: {state.current_question_index}")
        
    # Process "answer" action using both verifier and processor
    elif action.get("action") == "answer" and not state.is_complete:
        # First, verify the answer
        state = answer_verification_task(action, state, config=config).result()
        
        # Log after verification
        logger.debug(f"Parent-AfterVerification: conversation_history length: {len(state.conversation_history)}")
        logger.debug(f"Parent-AfterVerification: verification_result present: {bool(state.verification_result)}")
        
        # Then, process the verified answer if verification was completed
        if state.verification_result:
            # Pass the verification result to the question processor
            state = question_processor_task(state.verification_result, state, config=config).result()
            
            # Clear the verification_result after processing
            state.verification_result = {}
            
            # Log after processing
            logger.debug(f"Parent-AfterProcessing: conversation_history length: {len(state.conversation_history)}")
            logger.debug(f"Parent-AfterProcessing: current_question_index: {state.current_question_index}")
            
            # If there are items queued for term extraction, trigger extraction asynchronously
            if state.term_extraction_queue:
                logger.debug(f"Extraction: Triggering async term extraction. Queue: {state.term_extraction_queue}")
                thread = threading.Thread(
                    target=trigger_extraction_in_thread,
                    args=(thread_id,)
                )
                thread.daemon = True
                thread.start()
        else:
            logger.warning("Parent: No verification result found after verification task")
            
    # Process "extract_terms" action using the term extraction task
    elif action.get("action") == "extract_terms":
        # Call the term extraction task synchronously
        state = term_extraction_task(state, action, config=config).result()
        
        # Log key details after extraction
        logger.debug(f"Parent-AfterExtraction: conversation_history length: {len(state.conversation_history)}")
        logger.debug(f"Parent-AfterExtraction: current_question_index: {state.current_question_index}")
        logger.debug(f"Parent-AfterExtraction: responses count: {len(state.responses)}")
        logger.debug(f"Parent-AfterExtraction: term_extraction_queue: {state.term_extraction_queue}")
        
    # Process "status" action by collecting and logging system state
    elif action.get("action") == "status":
        # For status, we simply print relevant info
        logger.info("System Status:")
        logger.info(f"- Questions: {state.current_question_index}/{len(state.questions) if state.questions else 0}")
        logger.info(f"- Complete: {state.is_complete}")
        logger.info(f"- Extraction Queue: {len(state.term_extraction_queue)} items")
        logger.info(f"- Extracted Terms: {len(state.extracted_terms)} questions processed")
    
    # Log key details of final state
    logger.debug(f"Parent-Final: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"Parent-Final: current_question_index: {state.current_question_index}")
    logger.debug(f"Parent-Final: responses count: {len(state.responses)}")
    logger.debug(f"Parent-Final: term_extraction_queue: {state.term_extraction_queue}")
    
    return state

def trigger_extraction_in_thread(thread_id: Optional[str] = None):
    """Background thread function to trigger term extraction."""
    try:
        logger.debug(f"Thread: Starting extraction thread for thread_id: {thread_id}")
        time.sleep(0.5)  # Short delay
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        # Invoke the parent workflow with extract_terms action
        parent_workflow.invoke({"action": "extract_terms"}, config=config)
        logger.debug(f"Thread: Extraction completed for thread_id: {thread_id}")
    except Exception as e:
        import traceback
        logger.error(f"Thread: Error in extraction thread: {str(e)}")
        logger.error(traceback.format_exc())

def get_full_state(thread_id: str) -> Dict:
    """
    Retrieve the combined state from the parent workflow.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = parent_workflow.get_state(config)
        state = state_snapshot.values if hasattr(state_snapshot, "values") else {}
        result = {
            "current_index": state.get("current_question_index", 0),
            "is_complete": state.get("is_complete", False),
            "conversation_history": state.get("conversation_history", []),
            "responses": state.get("responses", {}),
            "extracted_terms": state.get("extracted_terms", {})
        }
        return result
    except Exception as e:
        import traceback
        logger.error(f"Error in get_full_state: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "current_index": 0,
            "is_complete": False,
            "conversation_history": [],
            "responses": {},
            "extracted_terms": {}
        }