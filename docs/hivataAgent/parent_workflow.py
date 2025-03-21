# parent_workflow.py - Coordinator for the multi-agent system

import os
import time
import json
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import the shared state
from state import ChatState, get_next_extraction_task

# Import the agent workflows
from question_answer import question_answer_workflow
from term_extractor import term_extraction_workflow, extraction_prompt, llm

# Define the main checkpointer for the parent workflow
parent_memory = MemorySaver()

@task
def process_term_extraction(state: ChatState, *, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Task to process items in the term extraction queue.
    
    Args:
        state: The current state
        config: Optional runtime configuration (now using keyword-only parameter)
        
    Returns:
        Updated state after term extraction
    """
    # Check if there are items in the queue
    if state.term_extraction_queue:
        print(f"Processing term extraction queue. Items: {len(state.term_extraction_queue)}")
        
        # Process the next item in the queue
        # We only process one item at a time to ensure sequential processing
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
        
        # Direct extraction approach instead of using the task
        if state.verified_answers:
            extracted_terms = {}
            for idx, answer_data in state.verified_answers.items():
                # Skip already processed items
                if idx in state.extracted_terms:
                    continue
                    
                question = answer_data.get("question", "")
                answer = answer_data.get("answer", "")
                
                # Format the prompt and extract terms
                formatted_prompt = extraction_prompt.format(
                    question=question,
                    answer=answer
                )
                
                print(f"Directly extracting terms for question {idx}: {question[:50]}...")
                
                # Use the LLM directly
                response = llm.invoke(formatted_prompt)
                content = response.content
                
                # Parse the results (same parsing as in direct-extract)
                try:
                    # Clean up the JSON response
                    content = content.strip()
                    if not content.startswith('['):
                        start_idx = content.find('[')
                        if start_idx >= 0:
                            content = content[start_idx:]
                            bracket_count = 0
                            for i, char in enumerate(content):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        content = content[:i+1]
                                        break
                    
                    terms = json.loads(content)
                    if isinstance(terms, list):
                        extracted_terms[idx] = terms
                        print(f"Extracted {len(terms)} terms for question {idx}")
                except Exception as e:
                    print(f"Error extracting terms: {str(e)}")
                    # Fallback approach
                    lines = content.split('\n')
                    terms = []
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith('-'):
                            terms.append(line[1:].strip())
                        elif line and line.startswith('"') and line.endswith('"'):
                            terms.append(line.strip('"'))
                    
                    extracted_terms[idx] = terms if terms else ["Error parsing terms"]
                    print(f"Fallback extraction: {len(extracted_terms[idx])} terms for question {idx}")
            
            # Update the extracted terms in the state
            if not hasattr(state, "extracted_terms") or state.extracted_terms is None:
                state.extracted_terms = {}
            state.extracted_terms.update(extracted_terms)
            print(f"Updated state with {len(extracted_terms)} sets of extracted terms")
    
    elif action.get("action") == "extract_terms":
        # Manually trigger term extraction
        state = process_term_extraction(state, config=config).result()
    
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
    
    try:
        # Get the parent state
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else {}
        
        # Get term extraction state for complete term information
        extract_state_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_state_snapshot.values if hasattr(extract_state_snapshot, "values") else {}
        
        # Extract terms could be in either parent state or extraction state
        extracted_terms = {}
        
        # Check parent state first
        if hasattr(parent_state, "extracted_terms") and parent_state.extracted_terms:
            extracted_terms.update(parent_state.extracted_terms)
        elif isinstance(parent_state, dict) and parent_state.get("extracted_terms"):
            extracted_terms.update(parent_state.get("extracted_terms", {}))
            
        # Then check extraction state
        if hasattr(extract_state, "extracted_terms") and extract_state.extracted_terms:
            extracted_terms.update(extract_state.extracted_terms)
        elif isinstance(extract_state, dict) and extract_state.get("extracted_terms"):
            extracted_terms.update(extract_state.get("extracted_terms", {}))
            
        # Combine the states
        result = {
            "current_index": parent_state.current_question_index if hasattr(parent_state, "current_question_index") else parent_state.get("current_question_index", 0),
            "is_complete": parent_state.is_complete if hasattr(parent_state, "is_complete") else parent_state.get("is_complete", False),
            "conversation_history": parent_state.conversation_history if hasattr(parent_state, "conversation_history") else parent_state.get("conversation_history", []),
            "responses": parent_state.responses if hasattr(parent_state, "responses") else parent_state.get("responses", {}),
            "extracted_terms": extracted_terms
        }
        
        return result
    except Exception as e:
        import traceback
        print(f"Error in get_full_state: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "current_index": 0,
            "is_complete": False,
            "conversation_history": [],
            "responses": {},
            "extracted_terms": {}
        }