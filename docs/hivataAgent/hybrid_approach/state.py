# state.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Import logger
from logging_config import logger

@dataclass
class SessionState:
    """Unified state for the entire workflow."""
    # Question-related state
    questions: List[Dict[str, Any]] = field(default_factory=list)
    current_index: int = 0
    is_complete: bool = False
    
    # Conversation tracking
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Response tracking
    responses: Dict[int, str] = field(default_factory=dict)
    verified_responses: Dict[int, bool] = field(default_factory=dict)
    
    # Verification state
    verification_messages: Dict[int, str] = field(default_factory=dict)
    verification_result: Dict[str, Any] = field(default_factory=dict)
    
    # For future extensions
    extracted_terms: Dict[int, List[str]] = field(default_factory=dict)
    term_extraction_queue: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Log when a new state instance is created."""
        logger.debug("New SessionState instance created")

def get_current_question(state: SessionState) -> Optional[Dict[str, Any]]:
    """Helper to get the current question from state."""
    if not state.questions or state.current_index >= len(state.questions):
        logger.debug(f"No current question available: index={state.current_index}, questions_count={len(state.questions)}")
        return None
    
    logger.debug(f"Retrieved current question (index={state.current_index})")
    return state.questions[state.current_index]

def format_question(question: Dict[str, Any]) -> str:
    """Format a question with its requirements."""
    logger.debug("Formatting question with requirements")
    
    if not question:
        logger.warning("Attempted to format an empty question")
        return ""
    
    try:
        formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in question.get("requirements", {}).items()])
        formatted_text = f"{question['text']}\n\nRequirements:\n{formatted_requirements}"
        logger.debug(f"Question formatted successfully: {formatted_text[:50]}...")
        return formatted_text
    except Exception as e:
        logger.error(f"Error formatting question: {str(e)}", exc_info=True)
        return "Error formatting question"

def get_formatted_current_question(state: SessionState) -> Optional[str]:
    """Get the current question formatted with requirements."""
    logger.debug(f"Getting formatted question for index {state.current_index}")
    
    question = get_current_question(state)
    if not question:
        logger.warning(f"No question found for index {state.current_index}")
        return None
    
    formatted = format_question(question)
    if not formatted:
        logger.warning("Failed to format question")
    
    return formatted

def add_to_extraction_queue(state: SessionState, question_index: int) -> SessionState:
    """Add a question index to the term extraction queue."""
    logger.info(f"Adding question {question_index} to term extraction queue")
    
    if question_index not in state.term_extraction_queue:
        state.term_extraction_queue.append(question_index)
        logger.debug(f"Question {question_index} added to extraction queue. Queue size: {len(state.term_extraction_queue)}")
    else:
        logger.debug(f"Question {question_index} already in extraction queue, skipping")
    
    return state

def convert_dict_keys_to_strings(data: Dict) -> Dict:
    """Convert all dictionary integer keys to strings for API compatibility."""
    logger.debug("Converting dictionary keys to strings")
    
    if not data:
        return {}
    
    result = {}
    for k, v in data.items():
        # Convert the key to string
        str_key = str(k)
        
        # Process the value recursively if it's a dictionary
        if isinstance(v, dict):
            result[str_key] = convert_dict_keys_to_strings(v)
        # Process list values if they contain dictionaries
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                result[str_key] = [convert_dict_keys_to_strings(item) if isinstance(item, dict) else item for item in v]
            else:
                result[str_key] = v
        else:
            result[str_key] = v
    
    return result

def to_dict(state: SessionState) -> Dict[str, Any]:
    """Convert SessionState to a dictionary for API responses."""
    logger.debug("Converting SessionState to dictionary for API response")
    
    try:
        # Create dictionary with all needed fields
        raw_dict = {
            "current_index": state.current_index,
            "is_complete": state.is_complete,
            "conversation_history": state.conversation_history,
            "responses": state.responses,
            "verified_responses": state.verified_responses,
            "verification_messages": state.verification_messages,
            "extracted_terms": state.extracted_terms,
            "current_question": get_formatted_current_question(state)
        }
        
        # Convert all integer keys to strings for FastAPI/Pydantic compatibility
        result = convert_dict_keys_to_strings(raw_dict)
        
        logger.debug(f"State converted to dict with {len(result)} keys")
        return result
    except Exception as e:
        logger.error(f"Error converting state to dict: {str(e)}", exc_info=True)
        # Return minimal information to avoid API failures
        return {
            "current_index": -1,
            "is_complete": False,
            "conversation_history": [],
            "error": "Error converting state to dictionary"
        }