# parent_workflow.py - Simplified using graph API and interrupt with accumulated history and improved logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage
from shared_memory import shared_memory
from logging_config import logger
from answer_verifier import verify_answer
from question_processor import get_questions
from term_extractor import extract_terms
from typing import Dict, Literal
from state import ChatState, init_state, PartialChatState
from utils import convert_messages, get_logging_context,format_question_prompt  # Shared utility for message conversion


# --- Node 1: init_node ---
def init_node(state: ChatState) -> ChatState:
    """
    Initializes the state with the questions and resets the conversation.
    Uses init_state() from state.py to build the base state.
    """
    try:
        logger.info("Initializing questionnaire state.", extra={"thread_id": state.get("thread_id")})
        questions = get_questions()
        logger.debug(f"Loaded {len(questions)} questions.", extra={"thread_id": state.get("thread_id")})
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}", extra={"thread_id": state.get("thread_id")})
        raise

    # Create the base state using the centralized initializer.
    new_state = init_state(state["thread_id"])
    # Populate the questions field.
    new_state["questions"] = questions
    return new_state

# --- Node 2: ask_node ---
def ask_node(state: ChatState) -> PartialChatState:
    idx = state["current_question_index"]
    context = get_logging_context(state, extra={"function": "ask_node"})
    try:
        logger.info(f"Preparing question index {idx}.", extra=context)
        if idx >= len(state["questions"]):
            logger.info("No more questions; marking session as complete.", extra=context)
            return {"is_complete": True}

        question_obj = state["questions"][idx]
        prompt = format_question_prompt(question_obj)
        logger.debug("Formatted question prompt.", extra=context)

        delta_history = [AIMessage(content=prompt)]
        logger.info(f"Returning delta with {len(delta_history)} new message(s).", extra=context)
    except Exception as e:
        logger.error(f"Error in ask_node for question index {idx}: {str(e)}", extra=context)
        raise

    if state.get("term_extraction_queue"):
        logger.info(f"Pending extraction tasks: {state['term_extraction_queue']}.", extra=context)
    return {"conversation_history": delta_history}

# --- Node 3: answer_node ---
def answer_node(state: ChatState) -> PartialChatState:
    idx = state["current_question_index"]
    context = get_logging_context(state, extra={"function": "answer_node"})
    try:
        logger.info(f"Waiting for user answer for question index {idx}.", extra=context)
        user_answer = interrupt({"prompt": "Please provide your answer:", "question_index": idx})
    except Exception as e:
        logger.error(f"Error during interrupt in answer_node for question index {idx}: {str(e)}", extra=context)
        raise
    logger.info(f"Received answer: {user_answer[:30]}...", extra=context)
    delta_history = [HumanMessage(content=user_answer)]
    logger.debug(f"Delta history length: {len(delta_history)}", extra=context)
    return {
        "conversation_history": delta_history,
        "responses": {**state["responses"], idx: user_answer}
    }

# --- Node 4: verification_node ---
def verification_node(state: ChatState) -> PartialChatState:
    idx = state["current_question_index"]
    context = get_logging_context(state, extra={"function": "verification_node"})
    try:
        logger.info(f"Processing verification for question index {idx}.", extra=context)
        conv_history_dicts = convert_messages(state["conversation_history"])
        logger.debug(f"Full conversation history length: {len(conv_history_dicts)}", extra=context)
    except Exception as e:
        logger.error(f"Error converting conversation history for question index {idx}: {str(e)}", extra=context)
        raise
    try:
        verification_result = verify_answer(state["questions"][idx], state["responses"].get(idx, ""), conv_history_dicts)
        verification_message = verification_result["verification_message"]
        is_valid = verification_result["is_valid"]
        logger.info(f"Verification outcome: is_valid={is_valid}.", extra=context)
    except Exception as e:
        logger.error(f"Error during verification for question index {idx}: {str(e)}", extra=context)
        raise

    delta_history = [AIMessage(content=verification_message)]
    logger.info("Appending verification message to conversation history.", extra=context)
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
def process_answer_node(state: ChatState) -> PartialChatState:
    try:
        idx = state["current_question_index"]
        context = get_logging_context(state, extra={"function": "process_answer_node"})
        logger.info(f"Processing answer for question index {idx}.", extra=context)
        
        verification_result = state.get("verification_result", {})
        if not verification_result or not verification_result.get("is_valid", False):
            logger.warning("Verification failed; re-prompting answer.", extra=context)
            return {}
        
        answer = verification_result.get("answer", "")
        verified_answers = {**state["verified_answers"]}
        verified_answers[idx] = {
            "question": state["questions"][idx]["text"],
            "answer": answer,
            "verification": verification_result.get("verification_message", "")
        }
        logger.debug(f"Updated verified_answers: {verified_answers}", extra=context)
        
        term_extraction_queue = state["term_extraction_queue"].copy()
        if idx not in term_extraction_queue:
            term_extraction_queue.append(idx)
            logger.debug(f"Added index {idx} to extraction queue: {term_extraction_queue}", extra=context)
            thread_state = {**state, "verified_answers": verified_answers, "term_extraction_queue": term_extraction_queue}
            if state.get("thread_id"):
                try:
                    import threading
                    logger.info(f"Launching background extraction for index {idx} with thread_id {state.get('thread_id')}.", extra=context)
                    extraction_thread = threading.Thread(
                        target=lambda: process_extraction_in_background(state.get("thread_id"), idx, shared_memory, current_state=thread_state),
                        daemon=True
                    )
                    extraction_thread.start()
                    logger.info(f"Started background extraction for index {idx}.", extra=context)
                except Exception as e:
                    logger.error(f"Error launching background extraction for index {idx}: {str(e)}", extra=context)
        
        new_index = idx + 1
        updates = {
            "current_question_index": new_index,
            "verified_answers": verified_answers,
            "term_extraction_queue": term_extraction_queue
        }
        
        if new_index >= len(state["questions"]):
            updates["is_complete"] = True
            delta_history = [AIMessage(content="Thank you for completing all the questions. Your responses have been recorded.")]
            updates["conversation_history"] = delta_history
            logger.info("All questions complete.", extra=context)
        
        # Recompute context after updating question index
        new_context = get_logging_context(state, extra={"function": "process_answer_node", "question_index": new_index})
        logger.info(f"Advancing to question index {new_index}.", extra=new_context)
        return updates

    except Exception as ex:
        logger.error(f"Unexpected error: {str(ex)}", extra=get_logging_context(state, extra={"function": "process_answer_node"}))
        raise

# --- Node 6: process_extraction_in_background ---
def process_extraction_in_background(thread_id: str, idx: int, memory_saver, current_state=None):
    """
    Process extraction in a background thread outside the graph flow.
    """
    context = {"thread_id": thread_id, "question_index": idx, "function": "process_extraction_in_background"}
    try:
        import time, copy
        time.sleep(0.5)
        logger.info(f"Background extraction started for index {idx}.", extra=context)

        if hasattr(memory_saver, "_states"):
            logger.debug(f"DEBUG: Memory saver states before extraction: {list(memory_saver._states.keys())}", extra={"thread_id": thread_id})
            if thread_id in memory_saver._states:
                logger.debug(f"DEBUG: thread_id {thread_id} exists in memory_saver._states", extra={"thread_id": thread_id})
        
        state_loaded = current_state if current_state is not None else memory_saver.load(thread_id)
        if not state_loaded:
            logger.error(f"No state found for thread_id {thread_id}.", extra=context)
            return
        logger.info(f"Background extraction using state with keys: {list(state_loaded.keys())}", extra=context)
        logger.debug(f"DEBUG: Initial extracted_terms: {state_loaded.get('extracted_terms')}", extra=context)
        
        if idx not in state_loaded.get("term_extraction_queue", []):
            logger.warning(f"Item {idx} not in extraction queue - queue: {state_loaded.get('term_extraction_queue', [])}", extra=context)
            return
        if idx not in state_loaded.get("verified_answers", {}):
            logger.error(f"No verified answer found for index {idx}.", extra=context)
            return
        
        verified_item = state_loaded["verified_answers"][idx]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        terms = extract_terms(question, answer)
        
        latest_state = memory_saver.load(thread_id)
        if not latest_state:
            logger.warning("Could not load latest state before update, using original state.", extra=context)
            latest_state = state_loaded
        
        new_queue = [i for i in latest_state.get("term_extraction_queue", []) if i != idx]
        current_terms = latest_state.get("extracted_terms", {})
        if not isinstance(current_terms, dict):
            logger.warning(f"extracted_terms is not a dictionary: {current_terms}. Creating new dictionary.", extra=context)
            current_terms = {}
        
        str_idx = str(idx)
        updated_terms = {str(k): v for k, v in current_terms.items()}
        updated_terms[str_idx] = terms
        
        updated_state = copy.deepcopy(latest_state)
        updated_state["term_extraction_queue"] = new_queue
        updated_state["extracted_terms"] = updated_terms
        updated_state["last_extracted_index"] = idx
        
        logger.debug(f"DEBUG: About to save state with keys: {list(updated_state.keys())}", extra=context)
        logger.debug(f"DEBUG: extracted_terms keys: {list(updated_state['extracted_terms'].keys())}", extra=context)
        logger.info(f"Saving extraction results for index {idx}: {terms[:3] if len(terms) > 3 else terms} (total: {len(terms)} terms)", extra=context)
        
        memory_saver.save(thread_id, updated_state)
        verification_state = memory_saver.load(thread_id)
        if verification_state:
            if "extracted_terms" in verification_state:
                extracted_terms = verification_state["extracted_terms"]
                logger.debug(f"DEBUG: Verification - extracted_terms keys: {list(extracted_terms.keys())}", extra=context)
                if str_idx in extracted_terms:
                    logger.info(f"Verified successful save of {len(extracted_terms[str_idx])} terms for index {idx}", extra=context)
                else:
                    logger.warning(f"Terms for index {idx} not found in saved state.", extra=context)
            else:
                logger.warning("No extracted_terms key in verification state.", extra=context)
        else:
            logger.error("Could not verify state save - failed to load for verification.", extra=context)
        
        logger.info(f"Background extraction completed for index {idx}, found {len(terms)} terms.", extra=context)
    except Exception as e:
        logger.error(f"Background extraction error at index {idx}: {str(e)}", extra=context)
        import traceback
        logger.error(traceback.format_exc(), extra=context)


# --- Node 7: extract_node ---
def extract_node(state: ChatState) -> PartialChatState:
    """
    Initiates asynchronous term extraction from verified answers.
    Processes items in the term extraction queue without blocking the main workflow.
    Delegates the actual extraction to a background thread.
    """
    thread_id = state.get("thread_id")
    context = get_logging_context(state, extra={"function": "extract_node"})
    try:
        logger.info("Processing term extraction queue.", extra=context)
        logger.info(f"Current term_extraction_queue: {state.get('term_extraction_queue')}", extra=context)
    except Exception as e:
        logger.error(f"Error logging extraction queue: {str(e)}", extra=context)
    
    if not state.get("term_extraction_queue"):
        logger.info("No items in extraction queue.", extra=context)
        return {}
    
    try:
        idx = state["term_extraction_queue"][0]
        context = get_logging_context(state, extra={"function": "extract_node", "question_index": idx})
        logger.info(f"Starting extraction for question index {idx}.", extra=context)
    except Exception as e:
        logger.error(f"Error retrieving index from extraction queue: {str(e)}", extra=context)
        return {}
    
    if idx in state.get("verified_answers", {}):
        logger.info(f"Found verified answer for index {idx}.", extra=context)
    else:
        logger.warning(f"No verified answer found for index {idx}. Removing index from queue.", extra=context)
        term_extraction_queue = [i for i in state.get("term_extraction_queue", []) if i != idx]
        return {"term_extraction_queue": term_extraction_queue}
    
    # Attempt thread_id retrieval via multiple methods for redundancy
    try:
        if thread_id:
            logger.info(f"Retrieved thread_id from state directly: {thread_id}", extra={"function": "extract_node", "question_index": idx})
        if not thread_id and hasattr(memory, 'parent_config') and hasattr(memory.parent_config, 'configurable'):
            thread_id = memory.parent_config.configurable.get('thread_id')
            logger.info(f"Retrieved thread_id from memory.parent_config: {thread_id}", extra={"question_index": idx})
        if not thread_id:
            try:
                if hasattr(state, 'config') and hasattr(state.config, 'configurable'):
                    thread_id = state.config.configurable.get('thread_id')
                    logger.info(f"Retrieved thread_id from state.config: {thread_id}", extra={"question_index": idx})
            except Exception as e:
                logger.error(f"Error getting thread_id from state.config: {str(e)}", extra={"question_index": idx})
        if not thread_id:
            try:
                if hasattr(memory, '_threads') and memory._threads:
                    latest_thread_id = list(memory._threads.keys())[-1]
                    thread_id = latest_thread_id
                    logger.info(f"Retrieved thread_id from memory._threads: {thread_id}", extra={"question_index": idx})
            except Exception as e:
                logger.error(f"Error getting thread_id from memory._threads: {str(e)}", extra={"question_index": idx})
    except Exception as e:
        logger.error(f"Error during thread_id retrieval: {str(e)}", extra=context)
    
    if not thread_id:
        logger.error("Cannot start async extraction - thread_id not available.", extra=context)
        logger.info("Falling back to synchronous extraction.", extra=context)
        return extract_node_sync(state)
    
    try:
        success = start_extraction_thread(state, idx, thread_id, memory)
    except Exception as e:
        logger.error(f"Exception when starting extraction thread for index {idx}: {str(e)}", extra={"thread_id": thread_id, "question_index": idx})
        success = False
    
    if success:
        logger.info(f"Async extraction started for index {idx}.", extra={"thread_id": thread_id, "question_index": idx})
        term_extraction_queue = [i for i in state.get("term_extraction_queue", []) if i != idx]
        return {"term_extraction_queue": term_extraction_queue}
    else:
        logger.error(f"Failed to start async extraction for index {idx}.", extra={"thread_id": thread_id, "question_index": idx})
        logger.info("Falling back to synchronous extraction.", extra={"thread_id": thread_id, "question_index": idx})
        return extract_node_sync(state)

# --- Alternate synchronous implementation ---
# This implementation processes extractions synchronously for testing or fallback
def extract_node_sync(state: ChatState) -> PartialChatState:
    """
    Synchronous term extraction processing - fallback implementation.
    """
    context = get_logging_context(state, extra={"function": "extract_node_sync"})
    logger.info("Running synchronous term extraction.", extra=context)
    
    if not state.get("term_extraction_queue"):
        return {}
    
    idx = state["term_extraction_queue"][0]
    context = get_logging_context(state, extra={"function": "extract_node_sync", "question_index": idx})
    logger.debug(f"Processing extraction for index {idx}.", extra=context)
    
    if idx in state.get("verified_answers", {}):
        verified_item = state["verified_answers"][idx]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        terms = extract_terms(question, answer)
        extracted_terms = {**state.get("extracted_terms", {})}
        term_extraction_queue = state.get("term_extraction_queue", []).copy()
        extracted_terms[idx] = terms
        try:
            term_extraction_queue.remove(idx)
        except Exception as e:
            logger.warning(f"Could not remove index {idx} from extraction queue: {str(e)}", 
                           extra=get_logging_context(state, extra={"function": "extract_node_sync", "question_index": idx}))
        logger.info(f"Extraction complete for index {idx}, found {len(terms)} terms.", extra=context)
        return {
            "extracted_terms": extracted_terms,
            "term_extraction_queue": term_extraction_queue,
            "last_extracted_index": idx
        }
    else:
        logger.warning(f"Index {idx} not found in verified_answers.", extra=context)
        return {}

# --- Node 7: end_node ---
def end_node(state: ChatState) -> Dict:
    """
    Final node that marks the questionnaire session as complete.
    """
    context = get_logging_context(state, extra={"function": "end_node"})
    logger.info("Finalizing state and marking session complete.", extra=context)
    return {"is_complete": True}

# --- Define routing logic ---
def verification_result_router(state: ChatState) -> Literal["process_answer_node", "answer_node"]:
    context = get_logging_context(state, extra={"function": "verification_result_router"})
    result = "process_answer_node" if state.get("verification_result", {}).get("is_valid", False) else "answer_node"
    logger.info(f"Routing to {result}.", extra=context)
    return result

def process_answer_router(state: ChatState) -> Literal["ask_node", "extract_node", "end_node"]:
    context = get_logging_context(state, extra={"function": "process_answer_router"})
    if state.get("trigger_extraction", False) and state.get("term_extraction_queue"):
        logger.info(f"Extraction trigger detected with queue: {state['term_extraction_queue']}", extra=context)
        state["trigger_extraction"] = False
        return "extract_node"
    if state["current_question_index"] >= len(state["questions"]):
        logger.info("No more questions, routing to end_node.", extra=context)
        if state.get("term_extraction_queue"):
            logger.info(f"Extraction queue present at end: {state['term_extraction_queue']}", extra=context)
            return "extract_node"
        return "end_node"
    logger.info(f"Proceeding to next question {state['current_question_index']}.", extra=context)
    return "ask_node"

def extract_router(state: ChatState) -> Literal["extract_node", "end_node"]:
    context = get_logging_context(state, extra={"function": "extract_router"})
    if state.get("term_extraction_queue"):
        logger.info(f"Routing to extract_node with queue: {state['term_extraction_queue']}", extra=context)
        return "extract_node"
    logger.info("No items in extraction queue, routing to end_node.", extra=context)
    return "end_node"

# --- Node 8: tasks_checker ---
def tasks_checker(state: ChatState) -> Dict:
    context = get_logging_context(state, extra={"function": "tasks_checker"})
    logger.info("Checking for pending background tasks.", extra=context)
    if state.get("term_extraction_queue"):
        logger.info(f"Found pending extraction tasks: {state['term_extraction_queue']}", extra=context)
    return {}

# --- Routing for task checking ---
def tasks_checker_router(state: ChatState) -> Literal["answer_node"]:
    return "answer_node"

# --- Node 9: trigger_extraction_node ---
def trigger_extraction_node(state: ChatState) -> Dict:
    context = get_logging_context(state, extra={"function": "trigger_extraction_node"})
    logger.info("Called to handle extraction trigger.", extra=context)
    if state.get("trigger_extraction", False):
        logger.info("Trigger detected, will process extraction queue.", extra=context)
        return {}
    else:
        logger.info("No trigger detected.", extra=context)
        return {}

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