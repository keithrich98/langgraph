# parent_workflow.py - Simplified using graph API and interrupt with accumulated history and improved logging
from typing import Dict, List, Literal, Any
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

# --- Node 1: init_node ---
def init_node(state: ChatState) -> Dict:
    """
    Initializes the state with the questions and resets the conversation.
    """
    logger.info("Initializing questionnaire state.")
    questions = get_questions()
    logger.debug(f"Loaded {len(questions)} questions.")
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
        "verification_result": {}
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

# --- Node 6: extract_node ---
def extract_node(state: ChatState) -> Dict:
    """
    A placeholder for term extraction functionality.
    """
    logger.info("extract_node: Running term extraction placeholder.")
    if not state["term_extraction_queue"]:
        return {}
    
    idx = state["term_extraction_queue"][0]
    logger.debug(f"extract_node: Processing extraction for index {idx}.")
    extracted_terms = {**state["extracted_terms"]}
    term_extraction_queue = state["term_extraction_queue"].copy()
    
    extracted_terms[idx] = ["placeholder term 1", "placeholder term 2"]
    term_extraction_queue.remove(idx)
    logger.info(f"extract_node: Extraction complete for index {idx}.")
    
    return {
        "extracted_terms": extracted_terms,
        "term_extraction_queue": term_extraction_queue,
        "last_extracted_index": idx
    }

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
    if state["current_question_index"] >= len(state["questions"]):
        if state["term_extraction_queue"]:
            return "extract_node"
        return "end_node"
    return "ask_node"

def extract_router(state: ChatState) -> Literal["extract_node", "end_node"]:
    """Route during extraction."""
    if state["term_extraction_queue"]:
        return "extract_node"
    return "end_node"

# --- Build the Graph ---
builder = StateGraph(ChatState)

# Add all nodes
builder.add_node("init_node", init_node)
builder.add_node("ask_node", ask_node)
builder.add_node("answer_node", answer_node)
builder.add_node("verification_node", verification_node)
builder.add_node("process_answer_node", process_answer_node)
builder.add_node("extract_node", extract_node)
builder.add_node("end_node", end_node)

# Set up edges
builder.add_edge(START, "init_node")
builder.add_edge("init_node", "ask_node")
builder.add_edge("ask_node", "answer_node")
builder.add_edge("answer_node", "verification_node")
builder.add_conditional_edges("verification_node", verification_result_router)
builder.add_conditional_edges("process_answer_node", process_answer_router)
builder.add_conditional_edges("extract_node", extract_router)
builder.add_edge("end_node", END)

# Create a checkpointer for persistence from shared_memory (already an instance)
memory = shared_memory

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=memory)
