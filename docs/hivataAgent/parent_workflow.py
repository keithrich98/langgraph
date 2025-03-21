# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import the shared state and helper
from state import ChatState, get_next_extraction_task

# Import the agent workflows
from question_answer import question_answer_workflow
from term_extractor import term_extraction_workflow

# Define the main checkpointer for the parent workflow
parent_memory = MemorySaver()

# ------------------------
# Updated Extraction Task
# ------------------------
@task
def process_term_extraction(state: ChatState, *, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Task to process one item from the term extraction queue.
    
    Args:
        state: The current state
        config: Optional runtime configuration (passed as keyword-only)
        
    Returns:
        Updated state after term extraction.
    """
    print(f"[DEBUG Extraction] process_term_extraction invoked. Current extraction queue: {state.term_extraction_queue}")
    if state.term_extraction_queue:
        print(f"[DEBUG Extraction] Extraction queue before processing: {state.term_extraction_queue}")
        extraction_result = term_extraction_workflow.invoke(
            {"action": "process", "state": state},  # Pass the parent's state in the payload
            config=config
        )
        print(f"[DEBUG Extraction] Extraction workflow returned state with queue: {extraction_result.term_extraction_queue} and extracted_terms: {extraction_result.extracted_terms}")
        if extraction_result:
            state.term_extraction_queue = extraction_result.term_extraction_queue
            state.extracted_terms = extraction_result.extracted_terms
            state.last_extracted_index = extraction_result.last_extracted_index
            print(f"[DEBUG Extraction] Merged state in process_term_extraction: extracted_terms: {state.extracted_terms}, queue: {state.term_extraction_queue}")
    else:
        print("[DEBUG Extraction] No items in extraction queue to process.")
    
    return state

# ------------------------
# Main Parent Workflow
# ------------------------
@entrypoint(checkpointer=parent_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Main workflow that coordinates the question_answer and term_extraction agents.
    
    Actions:
      {"action": "start"} - Start a new questionnaire session
      {"action": "answer", "answer": "user answer"} - Submit an answer to the current question
      {"action": "extract_terms"} - Manually trigger term extraction for the queue
      {"action": "status"} - Get the current status of the system
    
    This workflow automatically triggers term extraction after each valid answer.
    """
    # Use previous state or initialize a new one
    state = previous if previous is not None else ChatState()
    
    if action is None:
        # No action provided, return current state
        return state
    
    if action.get("action") in ["start", "answer"]:
        # Forward the action to the question_answer workflow
        qa_result = question_answer_workflow.invoke(action, config=config)
        
        if qa_result:
            # Update state fields from the QA workflow results
            state.current_question_index = qa_result.current_question_index
            state.questions = qa_result.questions if qa_result.questions else state.questions
            state.conversation_history = qa_result.conversation_history
            state.responses = qa_result.responses
            state.is_complete = qa_result.is_complete
            state.verified_answers = qa_result.verified_answers
            state.term_extraction_queue = qa_result.term_extraction_queue
            print(f"[DEBUG QA] Updated state from question_answer workflow. Extraction queue: {state.term_extraction_queue}")
        
        # Automatically process term extraction if items are in the queue
        if state.term_extraction_queue:
            print(f"[DEBUG Extraction] Extraction queue before processing: {state.term_extraction_queue}")
            state = process_term_extraction(state, config=config).result()
    
    elif action.get("action") == "extract_terms":
        # Manually trigger term extraction, ensuring proper config is passed
        print("[DEBUG Parent] Manually triggering term extraction.")
        state = process_term_extraction(state, config=config).result()
    
    elif action.get("action") == "status":
        # Retrieve status information from both workflows
        qa_state = question_answer_workflow.invoke({"action": "status"}, config=config)
        extract_state = term_extraction_workflow.invoke({"action": "status"}, config=config)
        
        print("System Status:")
        print(f"- Questions: {qa_state.current_question_index}/{len(qa_state.questions) if qa_state.questions else 0}")
        print(f"- Complete: {qa_state.is_complete}")
        print(f"- Extraction Queue: {len(extract_state.term_extraction_queue)} items")
        print(f"- Extracted Terms: {len(extract_state.extracted_terms)} questions processed")
    
    return state

# ------------------------
# Helper Function: Combined State
# ------------------------
def get_full_state(thread_id: str) -> Dict:
    """
    Get the full combined state for both agents.
    
    Args:
        thread_id: The thread ID
        
    Returns:
        Combined state as a dictionary.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get the parent state
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else {}
        
        # Get term extraction state for complete term information
        extract_state_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_state_snapshot.values if hasattr(extract_state_snapshot, "values") else {}
        
        # Combine the states into a single dictionary
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
