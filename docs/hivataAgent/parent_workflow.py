# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import the shared state
from state import ChatState, get_next_extraction_task

# Import the agent workflows
from question_answer import question_answer_workflow
from term_extractor import term_extraction_workflow

# Define the main checkpointer for the parent workflow
parent_memory = MemorySaver()

@task
def process_term_extraction(state: ChatState, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Task to process items in the term extraction queue.
    This runs the term extraction workflow for each item in the queue.
    
    Args:
        state: The current state
        config: Optional runtime configuration
        
    Returns:
        Updated state after term extraction
    """
    # Check if there are items in the queue
    if state.term_extraction_queue:
        print(f"Processing term extraction queue. Items: {len(state.term_extraction_queue)}")
        
        # Process the next item in the queue
        # We only process one item at a time to ensure sequential processing
        # Pass the exact same config to ensure thread_id consistency
        extraction_result = term_extraction_workflow.invoke(
            {"action": "process"}, 
            config=config  # Pass the parent config to maintain thread_id consistency
        )
        
        # Merge the extraction results back into our state
        if extraction_result:
            state.term_extraction_queue = extraction_result.term_extraction_queue
            state.extracted_terms = extraction_result.extracted_terms
            state.last_extracted_index = extraction_result.last_extracted_index
    
    return state

@entrypoint(checkpointer=parent_memory)
def parent_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Main workflow that coordinates the question_answer and term_extraction agents.
    
    Actions:
      {"action": "start"} - Start a new questionnaire session
      {"action": "answer", "answer": "user answer"} - Submit an answer to the current question
      {"action": "extract_terms"} - Manually trigger term extraction for the queue
      {"action": "status"} - Get the current status of the system
    
    The workflow automatically triggers term extraction after each valid answer.
    """
    # Use previous state or initialize a new one
    state = previous if previous is not None else ChatState()
    
    if action is None:
        # No action provided, just return current state
        return state
    
    if action.get("action") in ["start", "answer"]:
        # Forward to question_answer workflow
        qa_result = question_answer_workflow.invoke(action, config=config)
        
        # Update our state with the question_answer results
        if qa_result:
            # Copy over the relevant fields from the question_answer state
            state.current_question_index = qa_result.current_question_index
            state.questions = qa_result.questions if qa_result.questions else state.questions
            state.conversation_history = qa_result.conversation_history
            state.responses = qa_result.responses
            state.is_complete = qa_result.is_complete
            state.verified_answers = qa_result.verified_answers
            state.term_extraction_queue = qa_result.term_extraction_queue
        
        # Automatically process term extraction if there are items in the queue
        if state.term_extraction_queue:
            state = process_term_extraction(state, config).result()
    
    elif action.get("action") == "extract_terms":
        # Manually trigger term extraction
        state = process_term_extraction(state, config).result()
    
    elif action.get("action") == "status":
        # Get combined status of both agents
        qa_state = question_answer_workflow.invoke({"action": "status"}, config=config)
        extract_state = term_extraction_workflow.invoke({"action": "status"}, config=config)
        
        # We don't need to update the state, just print status information
        print(f"System Status:")
        print(f"- Questions: {qa_state.current_question_index}/{len(qa_state.questions) if qa_state.questions else 0}")
        print(f"- Complete: {qa_state.is_complete}")
        print(f"- Extraction Queue: {len(extract_state.term_extraction_queue)} items")
        print(f"- Extracted Terms: {len(extract_state.extracted_terms)} questions processed")
    
    return state

# Helper function to get the full combined state with all extracted terms
def get_full_state(thread_id: str) -> Dict:
    """
    Get the full combined state for both agents.
    
    Args:
        thread_id: The thread ID
        
    Returns:
        Combined state as a dictionary
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get the parent state
    parent_state = parent_workflow.get_state(config)
    
    # Get term extraction state for complete term information
    extract_state = term_extraction_workflow.get_state(config)
    
    # Combine the states
    result = {
        "current_index": parent_state.current_question_index if parent_state else 0,
        "is_complete": parent_state.is_complete if parent_state else False,
        "conversation_history": parent_state.conversation_history if parent_state else [],
        "responses": parent_state.responses if parent_state else {},
        "extracted_terms": extract_state.extracted_terms if extract_state else {}
    }
    
    return result