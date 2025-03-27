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
from question_answer import question_answer_task
from term_extractor import term_extraction_task
# Use the shared memory checkpointer
from shared_memory import shared_memory

def debug_state(prefix: str, state: ChatState):
    """Log key details of the state for debugging."""
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")  # to see if state objects are the same

@entrypoint(checkpointer=shared_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Parent workflow that coordinates question-answer and term extraction tasks.
    
    Supported actions:
      - {"action": "start"}: Initialize a new session.
      - {"action": "answer", "answer": "user answer"}: Process user answer.
      - {"action": "extract_terms"}: Process the next term extraction task.
      - {"action": "status"}: Report system status.
    """
    # Use previous state or initialize a new one.
    state = previous if previous is not None else ChatState()
    
    thread_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
    print(f"[DEBUG Parent] Running with thread_id: {thread_id}")
    debug_state("Parent-Initial", state)
    
    if action is None:
        return state

    # Process start or answer actions using the QA task.
    if action.get("action") in ["start", "answer"]:
        # Call the QA task synchronously passing current state and action.
        state = question_answer_task(action, state, config=config).result()
        debug_state("Parent-AfterQA", state)
        
        # For "answer", if there are items queued for term extraction, trigger extraction asynchronously.
        if action.get("action") == "answer" and state.term_extraction_queue:
            print(f"[DEBUG Extraction] Triggering async term extraction. Queue: {state.term_extraction_queue}")
            thread = threading.Thread(
                target=trigger_extraction_in_thread,
                args=(thread_id,)
            )
            thread.daemon = True
            thread.start()
            
    elif action.get("action") == "extract_terms":
        # Call the term extraction task synchronously.
        state = term_extraction_task(state, action, config=config).result()
        debug_state("Parent-AfterExtraction", state)
        
    elif action.get("action") == "status":
        # For status, we could simply print relevant info.
        print("System Status:")
        print(f"- Questions: {state.current_question_index}/{len(state.questions) if state.questions else 0}")
        print(f"- Complete: {state.is_complete}")
        print(f"- Extraction Queue: {len(state.term_extraction_queue)} items")
        print(f"- Extracted Terms: {len(state.extracted_terms)} questions processed")
    
    debug_state("Parent-Final", state)
    return state

def trigger_extraction_in_thread(thread_id: Optional[str] = None):
    """Background thread function to trigger term extraction."""
    try:
        print(f"[DEBUG Thread] Starting extraction thread for thread_id: {thread_id}")
        time.sleep(0.5)  # Short delay
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        # Invoke the parent workflow with extract_terms action.
        parent_workflow.invoke({"action": "extract_terms"}, config=config)
        print(f"[DEBUG Thread] Extraction completed for thread_id: {thread_id}")
    except Exception as e:
        import traceback
        print(f"[DEBUG Thread] Error in extraction thread: {str(e)}")
        print(traceback.format_exc())

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
