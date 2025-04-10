# parent_workflow.py - Simplified using graph API and interrupt with accumulated history and improved logging
from typing import Dict, List, Literal, Any, Optional
from typing_extensions import TypedDict  # Corrected import
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from shared_memory import shared_memory
from logging_config import logger
from answer_verifier import verify_answer
from question_processor import get_questions

# --- Define the ChatState Schema ---
class ChatState(TypedDict):
    # Core state fields
    current_question_index: int
    questions: List[Dict]
    # Use the add_messages reducer so that new updates are appended to the history.
    conversation_history: Annotated[List, add_messages]
    responses: Dict[int, str]
    is_complete: bool
    
    # Fields for verification and term extraction
    verified_answers: Dict[int, Dict[str, str]]
    term_extraction_queue: List[int]
    extracted_terms: Dict[int, List[str]]
    last_extracted_index: int
    verification_result: Dict[str, Any]
    
    # Configuration fields
    thread_id: Optional[str]  # Thread tracking - Do not use double underscore to avoid name mangling
    trigger_extraction: bool   # Flag to trigger extraction processing

# --- Node 1: init_node ---
def init_node(state: ChatState) -> Dict:
    """
    Initializes the state with the questions and resets the conversation.
    """
    logger.info("Initializing questionnaire state.")
    questions = get_questions()
    logger.debug(f"Loaded {len(questions)} questions.")
    
    # Get thread_id from state if it exists (should have been passed from API)
    thread_id = state.get("thread_id") if hasattr(state, "get") else None
    logger.debug(f"Initializing with thread_id: {thread_id}")
    
    # Start with an empty conversation history
    return {
        "questions": questions,
        "current_question_index": 0,
        "conversation_history": [],  # Empty; subsequent updates will be appended
        "responses": {},
        "is_complete": False,
        "verified_answers": {},
        "term_extraction_queue": [],
        "extracted_terms": {},
        "last_extracted_index": -1,
        "verification_result": {},
        "thread_id": thread_id,
        "trigger_extraction": False
    }

# --- Node 2: ask_node ---
def ask_node(state: ChatState) -> Dict:
    """
    Appends the current question (with requirements) to the conversation history.
    Returns only the new message so that the add_messages reducer appends it.
    """
    idx = state["current_question_index"]
    logger.info(f"ask_node: Preparing question index {idx}.")
    if idx >= len(state["questions"]):
        logger.info("No more questions; marking session as complete.")
        return {"is_complete": True}

    question_obj = state["questions"][idx]
    formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in question_obj["requirements"].items()])
    prompt = f"{question_obj['text']}\n\nRequirements:\n{formatted_requirements}"
    logger.debug("ask_node: Formatted question prompt.")
    
    # Return only the new AI message as the update.
    delta_history = [AIMessage(content=prompt)]
    logger.info(f"ask_node: Returning delta with {len(delta_history)} new message(s).")
    
    # Also log if there are pending extraction tasks, for debugging
    if state["term_extraction_queue"]:
        logger.info(f"ask_node: Note - there are pending extraction tasks: {state['term_extraction_queue']}")
    
    return {"conversation_history": delta_history}

# --- Node 4: answer_node ---
def answer_node(state: ChatState) -> dict:
    """
    Obtains the user's answer via an interrupt and returns only the new human message.
    """
    idx = state["current_question_index"]
    logger.info(f"answer_node: Waiting for user answer for question index {idx}.")
    
    # Note: The 'interrupt' function raises GraphInterrupt to pause the graph.
    user_answer = interrupt({"prompt": "Please provide your answer:", "question_index": idx})
    logger.info(f"answer_node: Received answer: {user_answer[:30]}...")
    
    # Return only the new human message.
    delta_history = [HumanMessage(content=user_answer)]
    logger.debug(f"answer_node: Delta history length: {len(delta_history)}")
    
    return {
        "conversation_history": delta_history,
        "responses": {**state["responses"], idx: user_answer}
    }

# --- Node 3: verification_node ---
def convert_messages(messages: list) -> list:
    """
    Converts a list of message objects to dictionaries.
    Takes any combination of dict messages, AIMessage, HumanMessage, etc.,
    and converts them all to a consistent dict format.
    """
    logger.debug(f"Converting {len(messages)} message(s) to dict format.")
    result = []
    for m in messages:
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            # Already in the right format
            result.append(m.copy())
        else:
            # Try various ways to extract role and content
            try:
                if hasattr(m, 'content'):
                    content = m.content
                    
                    # Determine the role
                    if hasattr(m, 'type'):
                        role = m.type.lower()
                    elif hasattr(m, '_message_type'):
                        msg_type = m._message_type.lower()
                        if 'ai' in msg_type or 'assistant' in msg_type:
                            role = 'ai'
                        elif 'human' in msg_type or 'user' in msg_type:
                            role = 'human'
                        else:
                            role = 'ai'  # Default fallback
                    else:
                        # Try to infer from class name
                        class_name = m.__class__.__name__.lower()
                        if 'ai' in class_name or 'assistant' in class_name:
                            role = 'ai'
                        elif 'human' in class_name or 'user' in class_name:
                            role = 'human'
                        else:
                            role = 'ai'  # Default fallback
                            
                    result.append({"role": role, "content": content})
                else:
                    logger.warning(f"Couldn't convert message to dict format: {m}")
            except Exception as e:
                logger.error(f"Error converting message: {e}")
                
    logger.debug(f"Converted messages: {result}")
    return result

def verification_node(state: ChatState) -> dict:
    """
    Verifies the user's answer using the LLM.
    Appends the verification message to the conversation history.
    Ensures that verification has access to the complete conversation history.
    """
    idx = state["current_question_index"]
    logger.info(f"verification_node: Processing verification for question index {idx}.")
    
    # Convert the FULL conversation history, not just deltas
    conv_history_dicts = convert_messages(state["conversation_history"])
    logger.debug(f"verification_node: Full conversation history length: {len(conv_history_dicts)}")
    
    verification_result = verify_answer(state["questions"][idx], state["responses"].get(idx, ""), conv_history_dicts)
    verification_message = verification_result["verification_message"]
    is_valid = verification_result["is_valid"]
    logger.info(f"verification_node: Verification outcome: is_valid={is_valid}.")
    
    # The add_messages reducer will append this to the existing history
    delta_history = [AIMessage(content=verification_message)]
    logger.info(f"verification_node: Appending verification message to conversation history.")
    
    return {
        "conversation_history": delta_history,
        "verification_result": {
            "action": "verified_answer",
            "question_index": idx,
            "answer": state["responses"].get(idx, ""),
            "verification_message": verification_message,
            "is_valid": is_valid
        }
    }

# --- Node 5: process_answer_node ---
def process_extraction_in_background(thread_id: str, idx: int, memory_saver, current_state=None):
    """
    Process extraction in a background thread outside the graph flow.
    """
    try:
        import time, copy
        time.sleep(0.5)
        
        logger.info(f"Background extraction started for index {idx}")
        
        if hasattr(memory_saver, "_states"):
            logger.debug(f"DEBUG: Memory saver states before extraction: {list(memory_saver._states.keys())}")
            if thread_id in memory_saver._states:
                logger.debug(f"DEBUG: thread_id {thread_id} exists in memory_saver._states")
                debug_state = memory_saver._states[thread_id]
                logger.debug(f"DEBUG: Direct state keys from _states: {list(debug_state.keys())}")
        
        state = current_state if current_state is not None else memory_saver.load(thread_id)
        
        if not state:
            logger.error(f"No state found for thread_id {thread_id}")
            return
            
        logger.info(f"Background extraction using state with keys: {list(state.keys())}")
        logger.debug(f"DEBUG: Initial extracted_terms: {state.get('extracted_terms')}")
        
        if idx not in state.get("term_extraction_queue", []):
            logger.warning(f"Item {idx} not in extraction queue - queue: {state.get('term_extraction_queue', [])}")
            return
            
        if idx not in state.get("verified_answers", {}):
            logger.error(f"No verified answer found for index {idx}")
            return
            
        verified_item = state["verified_answers"][idx]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        
        from term_extractor import extract_terms
        terms = extract_terms(question, answer)
        
        latest_state = memory_saver.load(thread_id)
        if not latest_state:
            logger.warning("Could not load latest state before update, using original state")
            latest_state = state
        
        new_queue = [i for i in latest_state.get("term_extraction_queue", []) if i != idx]
        current_terms = latest_state.get("extracted_terms", {})
        if not isinstance(current_terms, dict):
            logger.warning(f"extracted_terms is not a dictionary: {current_terms}, creating new dictionary")
            current_terms = {}
        
        str_idx = str(idx)
        updated_terms = {str(k): v for k, v in current_terms.items()}  # Ensure string keys
        
        updated_terms[str_idx] = terms
        
        updated_state = copy.deepcopy(latest_state)  # Use deep copy for the full state
        updated_state["term_extraction_queue"] = new_queue
        updated_state["extracted_terms"] = updated_terms
        updated_state["last_extracted_index"] = idx
        
        logger.debug(f"DEBUG: About to save state with keys: {list(updated_state.keys())}")
        logger.debug(f"DEBUG: extracted_terms keys: {list(updated_state['extracted_terms'].keys())}")
        logger.info(f"Saving extraction results for index {idx}: {terms[:3] if len(terms) > 3 else terms}... (total: {len(terms)} terms)")
        
        memory_saver.save(thread_id, updated_state)
        
        if hasattr(memory_saver, "_states") and thread_id in memory_saver._states:
            direct_state = memory_saver._states[thread_id]
            if "extracted_terms" in direct_state:
                direct_terms = direct_state["extracted_terms"]
                logger.debug(f"DEBUG: After save - direct _states access - extracted_terms keys: {list(direct_terms.keys())}")
                if str_idx in direct_terms:
                    logger.debug(f"DEBUG: Verified direct save - terms for index {str_idx} found in _states")
        
        verification_state = memory_saver.load(thread_id)
        if verification_state:
            if "extracted_terms" in verification_state:
                extracted_terms = verification_state["extracted_terms"]
                logger.debug(f"DEBUG: Verification - extracted_terms keys: {list(extracted_terms.keys())}")
                
                if str_idx in extracted_terms:
                    saved_terms = extracted_terms[str_idx]
                    logger.info(f"Verified successful save of {len(saved_terms)} terms for index {idx}")
                else:
                    logger.warning(f"Terms for index {idx} not found in saved state")
            else:
                logger.warning("No extracted_terms key in verification state")
        else:
            logger.error("Could not verify state save - failed to load for verification")
        
        logger.info(f"Background extraction completed for index {idx}, found {len(terms)} terms")
    except Exception as e:
        logger.error(f"Background extraction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def process_answer_node(state: ChatState) -> dict:
    """
    Processes the verified answer and advances to the next question.
    It returns updates (deltas) to the state and does not overwrite the accumulated history.
    """
    idx = state["current_question_index"]
    logger.info(f"process_answer_node: Processing answer for question index {idx}.")
    verification_result = state["verification_result"]
    if not verification_result or not verification_result.get("is_valid", False):
        logger.warning("process_answer_node: Verification failed; re-prompting answer.")
        return {}
    
    answer = verification_result.get("answer", "")
    verified_answers = {**state["verified_answers"]}
    verified_answers[idx] = {
        "question": state["questions"][idx]["text"],
        "answer": answer,
        "verification": verification_result.get("verification_message", "")
    }
    logger.debug(f"process_answer_node: Updated verified_answers: {verified_answers}")
    
    term_extraction_queue = state["term_extraction_queue"].copy()
    if idx not in term_extraction_queue:
        term_extraction_queue.append(idx)
        logger.debug(f"process_answer_node: Added index {idx} to extraction queue: {term_extraction_queue}")
        
        # Create the updated state with the extraction queue that we'll pass to the background thread
        thread_state = {**state}
        thread_state["verified_answers"] = verified_answers
        thread_state["term_extraction_queue"] = term_extraction_queue
        
        # Start background extraction immediately using threading
        thread_id = state.get("thread_id")
        if thread_id:
            import threading
            logger.info(f"process_answer_node: Launching background extraction with thread_id {thread_id} and state snapshot")
            extraction_thread = threading.Thread(
                target=lambda: process_extraction_in_background(thread_id, idx, shared_memory, current_state=thread_state),
                daemon=True
            )
            extraction_thread.start()
            logger.info(f"process_answer_node: Started background extraction for index {idx}")
    
    new_index = idx + 1
    updates = {
        "current_question_index": new_index,
        "verified_answers": verified_answers,
        "term_extraction_queue": term_extraction_queue
    }
    if new_index >= len(state["questions"]):
        updates["is_complete"] = True
        # Append a final AI message.
        delta_history = [AIMessage(content="Thank you for completing all the questions. Your responses have been recorded.")]
        updates["conversation_history"] = delta_history
        logger.info("process_answer_node: All questions complete.")
    
    logger.info(f"process_answer_node: Advancing to question index {new_index}.")
    return updates

# --- Import the term extractor ---
from term_extractor import start_extraction_thread

# --- Node 6: extract_node ---
def extract_node(state: ChatState) -> Dict:
    """
    Initiates asynchronous term extraction from verified answers.
    
    This node processes items in the term extraction queue without blocking the main workflow.
    It delegates the actual extraction to a background thread to keep the UI responsive.
    """
    logger.info("extract_node: Processing term extraction queue.")
    logger.info(f"extract_node: Current state term_extraction_queue: {state['term_extraction_queue']}")
    
    # Skip if queue is empty
    if not state["term_extraction_queue"]:
        logger.info("extract_node: No items in extraction queue.")
        return {}
    
    # Get the first item from the queue
    idx = state["term_extraction_queue"][0]
    logger.info(f"extract_node: Starting extraction for question index {idx}.")
    
    # For debugging: Check if the item is in verified_answers
    if idx in state["verified_answers"]:
        logger.info(f"extract_node: Found verified answer for index {idx}.")
    else:
        logger.warning(f"extract_node: No verified answer found for index {idx}.")
        # Remove from queue and return
        term_extraction_queue = [i for i in state["term_extraction_queue"] if i != idx]
        return {"term_extraction_queue": term_extraction_queue}
    
    # ENHANCED: Try multiple ways to get thread_id
    thread_id = None
    
    # 1. First, check if thread_id is stored directly in state (new approach)
    if "thread_id" in state:
        thread_id = state["thread_id"]
        logger.info(f"extract_node: Retrieved thread_id from state directly: {thread_id}")
    
    # 2. Try to get thread_id from memory's parent_config (original approach)
    if not thread_id and hasattr(memory, 'parent_config') and hasattr(memory.parent_config, 'configurable'):
        thread_id = memory.parent_config.configurable.get('thread_id')
        logger.info(f"extract_node: Retrieved thread_id from memory.parent_config: {thread_id}")
    
    # 3. Try to get thread_id from state's config attribute (original approach)
    if not thread_id:
        try:
            if hasattr(state, 'config') and hasattr(state.config, 'configurable'):
                thread_id = state.config.configurable.get('thread_id')
                logger.info(f"extract_node: Retrieved thread_id from state.config: {thread_id}")
        except Exception as e:
            logger.error(f"extract_node: Error getting thread_id from state.config: {str(e)}")
    
    # 4. Try to get thread_id from memory's latest threads (new approach)
    if not thread_id:
        try:
            # This is a hacky approach but might work - check if memory has a _threads attribute
            if hasattr(memory, '_threads') and memory._threads:
                # Get the most recent thread_id
                latest_thread_id = list(memory._threads.keys())[-1]
                thread_id = latest_thread_id
                logger.info(f"extract_node: Retrieved thread_id from memory._threads: {thread_id}")
        except Exception as e:
            logger.error(f"extract_node: Error getting thread_id from memory._threads: {str(e)}")
    
    if not thread_id:
        logger.error("extract_node: Cannot start async extraction - thread_id not available")
        # We'll use synchronous extraction as a fallback
        logger.info("extract_node: Falling back to synchronous extraction")
        return extract_node_sync(state)
    
    # Start asynchronous extraction in a separate thread
    success = start_extraction_thread(state, idx, thread_id, memory)
    
    if success:
        logger.info(f"extract_node: Async extraction started for index {idx}.")
        # Remove item from the queue immediately to avoid re-processing
        # The extraction thread will update the results in the state directly
        term_extraction_queue = [i for i in state["term_extraction_queue"] if i != idx]
        return {"term_extraction_queue": term_extraction_queue}
    else:
        logger.error(f"extract_node: Failed to start async extraction for index {idx}.")
        # Try synchronous extraction as a fallback
        logger.info("extract_node: Falling back to synchronous extraction")
        return extract_node_sync(state)

# --- Alternate synchronous implementation ---
# This implementation processes extractions synchronously for testing or fallback
def extract_node_sync(state: ChatState) -> Dict:
    """
    Synchronous term extraction processing - can be used as a fallback.
    """
    from term_extractor import extract_terms
    
    logger.info("extract_node_sync: Running synchronous term extraction.")
    if not state["term_extraction_queue"]:
        return {}
    
    idx = state["term_extraction_queue"][0]
    logger.debug(f"extract_node_sync: Processing extraction for index {idx}.")
    
    # Get the question and answer
    if idx in state["verified_answers"]:
        verified_item = state["verified_answers"][idx]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        
        # Extract terms synchronously
        terms = extract_terms(question, answer)
        
        # Update state
        extracted_terms = {**state["extracted_terms"]}
        term_extraction_queue = state["term_extraction_queue"].copy()
        
        extracted_terms[idx] = terms
        term_extraction_queue.remove(idx)
        logger.info(f"extract_node_sync: Extraction complete for index {idx}, found {len(terms)} terms.")
        
        return {
            "extracted_terms": extracted_terms,
            "term_extraction_queue": term_extraction_queue,
            "last_extracted_index": idx
        }
    else:
        logger.warning(f"extract_node_sync: Index {idx} not found in verified_answers.")
        return {}

# --- Node 7: end_node ---
def end_node(state: ChatState) -> Dict:
    """
    Final node that marks the questionnaire session as complete.
    """
    logger.info("end_node: Finalizing state and marking session complete.")
    return {"is_complete": True}

# --- Define routing logic ---
def verification_result_router(state: ChatState) -> Literal["process_answer_node", "answer_node"]:
    """Route based on verification result."""
    verification_result = state["verification_result"]
    if verification_result and verification_result.get("is_valid", False):
        return "process_answer_node"
    return "answer_node"

def process_answer_router(state: ChatState) -> Literal["ask_node", "extract_node", "end_node"]:
    """Route after processing an answer."""
    # Check if we have a trigger to explicitly process extractions
    if state.get("trigger_extraction", False) and state["term_extraction_queue"]:
        logger.info(f"process_answer_router: Extraction trigger detected with queue: {state['term_extraction_queue']}")
        # Reset the trigger flag
        state["trigger_extraction"] = False
        return "extract_node"
    
    # First check if we've reached the end of questions
    if state["current_question_index"] >= len(state["questions"]):
        logger.info("process_answer_router: No more questions, routing to end_node")
        # If we're at the end and still have extraction tasks, route to extraction
        if state["term_extraction_queue"]:
            logger.info(f"process_answer_router: At end of questions with extraction queue: {state['term_extraction_queue']}")
            return "extract_node"
        return "end_node"
    
    # If we have more questions, prioritize going to the next question over extraction
    # This ensures the user can continue with the next question while extraction happens in background
    logger.info(f"process_answer_router: Proceeding to next question {state['current_question_index']}")
    return "ask_node"

def extract_router(state: ChatState) -> Literal["extract_node", "end_node"]:
    """Route during extraction."""
    # Add debug logging to show the routing decision
    if state["term_extraction_queue"]:
        logger.info(f"extract_router: Routing to extract_node with queue: {state['term_extraction_queue']}")
        return "extract_node"
    logger.info("extract_router: No items in extraction queue, routing to end_node")
    return "end_node"

# --- Node 8: tasks_checker ---
def tasks_checker(state: ChatState) -> Dict:
    """
    A utility node to check and potentially start background tasks.
    This helps ensure that extraction tasks can be processed even
    when the main flow prioritizes asking the next question.
    """
    logger.info("tasks_checker: Checking for pending background tasks.")
    
    # Check if there are pending extraction tasks
    if state["term_extraction_queue"]:
        logger.info(f"tasks_checker: Found pending extraction tasks: {state['term_extraction_queue']}")
        
        # We won't automatically start extraction here, but this node
        # provides a hook for potentially handling background tasks.
        # For now, we just log the detection.
        
        # In the future, we could add logic to start extraction as a true
        # background process from this node
    
    # No direct state changes, this node is mainly for logging/monitoring
    return {}

# --- Node 9: trigger_extraction_node ---
def trigger_extraction_node(state: ChatState) -> Dict:
    """
    Handle the trigger_extraction flag and route to process extractions.
    This node is called when the API's /terms endpoint triggers extraction processing.
    """
    logger.info("trigger_extraction_node: Called to handle extraction trigger")
    
    if state.get("trigger_extraction", False):
        logger.info("trigger_extraction_node: Trigger detected, will process extraction queue")
        # Pass through the trigger flag, don't reset it here
        return {}
    else:
        logger.info("trigger_extraction_node: No trigger detected")
        return {}

# --- Routing for task checking ---
def tasks_checker_router(state: ChatState) -> Literal["answer_node"]:
    """Always route back to answer_node after task checking."""
    # The tasks_checker is just a monitoring node, we always continue
    # with the user interaction flow
    return "answer_node"

# --- Routing for trigger extraction ---
def trigger_extraction_router(state: ChatState) -> Literal["process_answer_node"]:
    """Always route to process_answer_node to handle the trigger."""
    return "process_answer_node"

# --- Build the Graph ---
builder = StateGraph(ChatState)

# Add all nodes
builder.add_node("init_node", init_node)
builder.add_node("ask_node", ask_node)
builder.add_node("tasks_checker", tasks_checker)  # Monitoring node
builder.add_node("answer_node", answer_node)
builder.add_node("verification_node", verification_node)
builder.add_node("process_answer_node", process_answer_node)
builder.add_node("extract_node", extract_node)
builder.add_node("trigger_extraction_node", trigger_extraction_node)  # New trigger node
builder.add_node("end_node", end_node)

# Set up edges
builder.add_edge(START, "init_node")
builder.add_edge("init_node", "ask_node")
# Insert tasks_checker between ask_node and answer_node to log pending tasks
builder.add_edge("ask_node", "tasks_checker")
builder.add_conditional_edges("tasks_checker", tasks_checker_router)
builder.add_edge("answer_node", "verification_node")
builder.add_conditional_edges("verification_node", verification_result_router)
# Add direct entry point for extraction triggering
builder.add_edge("trigger_extraction_node", "process_answer_node")
builder.add_conditional_edges("process_answer_node", process_answer_router)
builder.add_conditional_edges("extract_node", extract_router)
builder.add_edge("end_node", END)

# Create a checkpointer for persistence from shared_memory (already an instance)
memory = shared_memory

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=memory)
