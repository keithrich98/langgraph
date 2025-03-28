# state.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import threading
from logging_config import logger

logger.info("Initializing state.py")

@dataclass
class ChatState:
    """
    Shared state for the medical questionnaire system.
    """
    # Current question index in the questionnaire
    current_question_index: int = 0
    
    # List of questions with their requirements
    questions: List[Dict] = field(default_factory=list)
    
    # Full conversation history
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Raw responses from the user, mapped by question index
    responses: Dict[int, str] = field(default_factory=dict)
    
    # Flag indicating if the questionnaire is complete
    is_complete: bool = False
    
    # Storage for verified answers
    verified_answers: Dict[int, Dict[str, str]] = field(default_factory=dict)

# Thread-safe storage for states
_state_storage = {}
_state_lock = threading.Lock()

def get_state_for_thread(thread_id: str) -> Optional[ChatState]:
    """
    Get the state for a specific thread ID.
    """
    logger.info(f"get_state_for_thread called for thread_id: {thread_id}")
    with _state_lock:
        state = _state_storage.get(thread_id)
        if state:
            logger.debug(f"Found state for thread_id: {thread_id}")
            logger.debug(f"State details - question index: {state.current_question_index}, is_complete: {state.is_complete}")
            logger.debug(f"Conversation history length: {len(state.conversation_history)}")
        else:
            logger.debug(f"No state found for thread_id: {thread_id}")
        return state

def update_state_for_thread(thread_id: str, state: ChatState) -> None:
    """
    Update the state for a specific thread ID.
    """
    logger.info(f"update_state_for_thread called for thread_id: {thread_id}")
    logger.debug(f"Updating state - question index: {state.current_question_index}, is_complete: {state.is_complete}")
    logger.debug(f"Conversation history length: {len(state.conversation_history)}")
    with _state_lock:
        _state_storage[thread_id] = state
        logger.debug(f"State updated for thread_id: {thread_id}")

def remove_state_for_thread(thread_id: str) -> None:
    """
    Remove the state for a specific thread ID.
    """
    logger.info(f"remove_state_for_thread called for thread_id: {thread_id}")
    with _state_lock:
        if thread_id in _state_storage:
            del _state_storage[thread_id]
            logger.debug(f"State removed for thread_id: {thread_id}")
        else:
            logger.debug(f"No state found to remove for thread_id: {thread_id}")