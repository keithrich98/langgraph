# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
import threading
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import the shared state and helper
from state import ChatState, get_next_extraction_task

# Import the agent workflows
from question_answer import question_answer_workflow
from term_extractor import term_extraction_workflow

# Define the main checkpointer for the parent workflow.
# This checkpointer is used to persist the parent state between invocations.
parent_memory = MemorySaver()

# ------------------------
# Main Parent Workflow
# ------------------------
@entrypoint(checkpointer=parent_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    This is the main workflow that coordinates the question_answer and term_extraction agents.
    
    The workflow supports multiple actions:
    
      - {"action": "start"}:
          * Initialize a new questionnaire session.
          * The question_answer workflow sends the first question to the human.
          
      - {"action": "answer", "answer": "user answer"}:
          * The human submits an answer.
          * The question_answer workflow verifies the answer.
          * If verified, the verified answer is stored in state.verified_answers and its index is added to the term_extraction_queue.
          * The workflow then triggers term extraction asynchronously in a background thread.
          
      - {"action": "extract_terms"}:
          * Process items in the term extraction queue.
          
      - {"action": "status"}:
          * Retrieve and print the current status from both the question_answer and term_extraction workflows.
    
    After the question_answer workflow updates the state with the new answer (and queues it for extraction),
    if there are items in state.term_extraction_queue, the workflow starts a background thread to handle extraction.
    This allows the API to return the next question immediately without waiting for term extraction to complete.
    """
    # Use previous state or initialize a new one.
    state = previous if previous is not None else ChatState()
    
    if action is None:
        # If no action is provided, simply return the current state.
        return state
    
    if action.get("action") in ["start", "answer"]:
        # Forward the action (start or answer) to the question_answer workflow.
        qa_result = question_answer_workflow.invoke(action, config=config)
        
        if qa_result:
            # Update the parent's state with the fields from the question_answer workflow:
            # This includes the current question index, questions list, conversation history,
            # responses, completion status, verified answers, and the extraction queue.
            state.current_question_index = qa_result.current_question_index
            state.questions = qa_result.questions if qa_result.questions else state.questions
            state.conversation_history = qa_result.conversation_history
            state.responses = qa_result.responses
            state.is_complete = qa_result.is_complete
            state.verified_answers = qa_result.verified_answers
            state.term_extraction_queue = qa_result.term_extraction_queue
            print(f"[DEBUG QA] Updated state from question_answer workflow. Extraction queue: {state.term_extraction_queue}")
        
        # For answer actions, trigger term extraction asynchronously if there are items in the queue
        if action.get("action") == "answer" and state.term_extraction_queue:
            print(f"[DEBUG Extraction] Triggering async term extraction. Queue: {state.term_extraction_queue}")
            
            # Extract just the thread_id from the config, don't try to copy the whole config
            thread_id = config.get("configurable", {}).get("thread_id") if config else None
            
            # Start the extraction in a background thread
            thread = threading.Thread(
                target=trigger_extraction_in_thread,
                args=(thread_id,)
            )
            thread.daemon = True  # Set as daemon so it doesn't block program exit
            thread.start()
    
    elif action.get("action") == "extract_terms":
        # This is the action that runs the term extraction
        print("[DEBUG Parent] Running term extraction.")
        
        if state.term_extraction_queue:
            print(f"[DEBUG Extraction] Processing extraction queue: {state.term_extraction_queue}")
            # Invoke term_extraction_workflow to process the next item in the queue
            next_index = get_next_extraction_task(state)
            if next_index is not None and next_index in state.verified_answers:
                extraction_result = term_extraction_workflow.invoke(
                    {"action": "process", "state": state},
                    config=config
                )
                
                if extraction_result:
                    # Update extraction-related state fields
                    state.term_extraction_queue = extraction_result.term_extraction_queue
                    state.extracted_terms = extraction_result.extracted_terms
                    state.last_extracted_index = extraction_result.last_extracted_index
                    print(f"[DEBUG Extraction] Updated state after extraction: queue={state.term_extraction_queue}, extracted_terms keys={list(state.extracted_terms.keys() if state.extracted_terms else [])}")
        else:
            print("[DEBUG Extraction] No items in extraction queue to process.")
    
    elif action.get("action") == "status":
        # For status, invoke both the question_answer and term_extraction workflows with action "status"
        qa_state = question_answer_workflow.invoke({"action": "status"}, config=config)
        extract_state = term_extraction_workflow.invoke({"action": "status"}, config=config)
        
        print("System Status:")
        print(f"- Questions: {qa_state.current_question_index}/{len(qa_state.questions) if qa_state.questions else 0}")
        print(f"- Complete: {qa_state.is_complete}")
        print(f"- Extraction Queue: {len(extract_state.term_extraction_queue)} items")
        print(f"- Extracted Terms: {len(extract_state.extracted_terms)} questions processed")
    
    return state

# Add this new function to trigger extraction from a thread
def trigger_extraction_in_thread(thread_id: Optional[str] = None):
    """
    Thread target function that invokes the parent_workflow with the extract_terms action.
    This properly preserves the LangGraph context.
    
    Args:
        thread_id: The thread ID to use in the config.
    """
    try:
        print(f"[DEBUG Thread] Starting background extraction thread for thread_id: {thread_id}")
        # Short delay to ensure the main thread has completed its operation
        time.sleep(0.5)
        
        # Create a minimal config with just the thread_id
        thread_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        
        # Invoke the parent workflow with the extract_terms action
        parent_workflow.invoke({"action": "extract_terms"}, config=thread_config)
        print(f"[DEBUG Thread] Extraction completed successfully for thread_id: {thread_id}")
    except Exception as e:
        import traceback
        print(f"[DEBUG Thread] Error in extraction thread: {str(e)}")
        print(traceback.format_exc())

# ------------------------
# Helper Function: Combined State
# ------------------------
def get_full_state(thread_id: str) -> Dict:
    """
    Retrieve the full combined state from both the parent workflow and the term extraction workflow.
    
    This function uses the parent's checkpoint to get the current state, and separately retrieves
    the term extraction state (which should have the extracted_terms field updated by the term_extraction_workflow).
    The two are merged into a single dictionary.
    
    Args:
        thread_id: The thread identifier (session id).
        
    Returns:
        A dictionary representing the combined state.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Retrieve the parent state snapshot.
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else {}
        
        # Retrieve the term extraction workflow state snapshot.
        extract_state_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_state_snapshot.values if hasattr(extract_state_snapshot, "values") else {}
        
        # Combine the states into a single dictionary.
        result = {
            "current_index": parent_state.get("current_question_index", 0),
            "is_complete": parent_state.get("is_complete", False),
            "conversation_history": parent_state.get("conversation_history", []),
            "responses": parent_state.get("responses", {}),
            "extracted_terms": extract_state.get("extracted_terms", {})
        }
        
        return result
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in get_full_state: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "current_index": 0,
            "is_complete": False,
            "conversation_history": [],
            "responses": {},
            "extracted_terms": {}
        }