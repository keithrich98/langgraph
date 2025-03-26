# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
import threading
from typing import Dict, Any, Optional
from langgraph.func import entrypoint, task
from langchain_core.runnables import RunnableConfig

from state import ChatState, get_next_extraction_task
from shared_memory import shared_memory

# Import the child tasks instead of entrypoints.
from question_answer import process_question_answer
from term_extractor import process_term_extraction

def debug_state(prefix: str, state: ChatState):
    """Log the key details of the state for debugging."""
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")  # Helps identify if state objects are the same

@entrypoint(checkpointer=shared_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Main workflow that coordinates the question-answer and term-extraction tasks.
    
    Actions:
      - {"action": "start"}: Initialize a new session.
      - {"action": "answer", "answer": "user answer"}: Process a new answer.
      - {"action": "extract_terms"}: Process term extraction.
      - {"action": "status"}: Log the current state.
      
    This refactored workflow calls child agents as tasks so that the parent state is always passed along.
    """
    # Use previous state or initialize a new one.
    state = previous if previous is not None else ChatState()
    
    thread_id = config.get("configurable", {}).get("thread_id") if config else "unknown"
    print(f"[DEBUG Parent] Running with thread_id: {thread_id}")
    debug_state("Parent-Initial", state)
    
    if action is None:
        return state

    if action.get("action") in ["start", "answer"]:
        # Process the question-answer task. Pass the current state and action.
        state = process_question_answer(state, action, config=config).result()
        print(f"[DEBUG Parent] After QA task:")
        debug_state("Parent-AfterQA", state)
        
        # If the action was "answer" and there are items in the extraction queue, process term extraction.
        if action.get("action") == "answer" and state.term_extraction_queue:
            print(f"[DEBUG Extraction] Triggering term extraction task. Queue: {state.term_extraction_queue}")
            state = process_term_extraction(state, config=config).result()
            print(f"[DEBUG Parent] After Term Extraction task:")
            debug_state("Parent-AfterExtraction", state)
    
    elif action.get("action") == "extract_terms":
        print("[DEBUG Parent] Running term extraction task explicitly.")
        if state.term_extraction_queue:
            state = process_term_extraction(state, config=config).result()
            debug_state("Parent-AfterExtraction", state)
        else:
            print("[DEBUG Extraction] No items in extraction queue to process.")
    
    elif action.get("action") == "status":
        print("System Status:")
        print(f"- Questions: {state.current_question_index}/{len(state.questions)}")
        print(f"- Complete: {state.is_complete}")
        print(f"- Extraction Queue: {len(state.term_extraction_queue)} items")
    
    debug_state("Parent-Final", state)
    return state

# Optional helper for asynchronous extraction triggering.
def trigger_extraction_in_thread(thread_id: Optional[str] = None):
    """
    Thread target function that triggers term extraction by invoking the parent_workflow.
    """
    try:
        print(f"[DEBUG Thread] Starting background extraction thread for thread_id: {thread_id}")
        time.sleep(0.5)  # Ensure main thread has completed its operation.
        thread_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        parent_workflow.invoke({"action": "extract_terms"}, config=thread_config)
        print(f"[DEBUG Thread] Extraction completed successfully for thread_id: {thread_id}")
    except Exception as e:
        import traceback
        print(f"[DEBUG Thread] Error in extraction thread: {str(e)}")
        print(traceback.format_exc())

def get_full_state(thread_id: str) -> Dict:
    """
    Retrieve the full combined state using the parent's checkpoint.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else {}
        result = {
            "current_index": parent_state.get("current_question_index", 0),
            "is_complete": parent_state.get("is_complete", False),
            "conversation_history": parent_state.get("conversation_history", []),
            "responses": parent_state.get("responses", {}),
            "extracted_terms": parent_state.get("extracted_terms", {})
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
