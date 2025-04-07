# state.py
from typing import Dict, List, Any, Optional, TypedDict, Union
from pydantic import BaseModel, Field
from copy import deepcopy

# Import logger
from hivataAgent.hybrid_approach.config.logging_config import logger

class SessionState(BaseModel):
    """
    Unified state for the entire workflow using Pydantic for better LangGraph integration.
    
    Uses Pydantic's immutable model pattern for thread safety when using asyncio or threading.
    Use the model_copy or update() method to create a new state instance with changes.
    """
    # Question-related state
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    current_index: int = 0
    is_complete: bool = False
    
    # Current action being processed
    current_action: Optional[Dict[str, Any]] = None
    
    # Conversation tracking
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # Response tracking
    responses: Dict[int, str] = Field(default_factory=dict)
    verified_responses: Dict[int, bool] = Field(default_factory=dict)
    
    # Verification state
    verification_messages: Dict[int, str] = Field(default_factory=dict)
    verification_result: Dict[str, Any] = Field(default_factory=dict)
    
    # Storage for verified answers ready for term extraction
    # Format: {question_index: {"question": str, "answer": str, "verification": str}}
    verified_answers: Dict[int, Dict[str, str]] = Field(default_factory=dict)
    
    # Term extraction state
    extracted_terms: Dict[int, List[str]] = Field(default_factory=dict)
    term_extraction_queue: List[int] = Field(default_factory=list)
    
    # Error tracking
    error: Optional[str] = None
    
    model_config = {
        # Prevent arbitrary attributes from being set
        "extra": "forbid",
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        logger.debug("New SessionState instance created")
    
    def update(self, **updates) -> "SessionState":
        """
        Create a new state instance with updates applied.
        
        This method ensures immutability by creating a new copy with specified updates.
        For nested dictionaries and lists, deep copies are created to prevent shared references.
        
        Args:
            **updates: Keyword arguments with fields to update
            
        Returns:
            A new SessionState instance with updates applied
        """
        # Handle nested structures that need deep copying
        processed_updates = {}
        
        for key, value in updates.items():
            if isinstance(value, dict) or isinstance(value, list):
                processed_updates[key] = deepcopy(value)
            else:
                processed_updates[key] = value
        
        # Create a new instance with updates
        try:
            new_state = self.model_copy(update=processed_updates)
            logger.debug(f"Created updated state with changes to: {', '.join(updates.keys())}")
            return new_state
        except Exception as e:
            logger.error(f"Error creating updated state: {str(e)}", exc_info=True)
            # Return original state if update fails
            return self

# Helper functions that work with SessionState

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
    
    # Avoid modifying the original state
    if question_index not in state.term_extraction_queue:
        new_queue = state.term_extraction_queue.copy() + [question_index]
        logger.debug(f"Question {question_index} added to extraction queue. Queue size: {len(new_queue)}")
        return state.update(term_extraction_queue=new_queue)
    else:
        logger.debug(f"Question {question_index} already in extraction queue, not modifying state")
        return state

def get_next_extraction_task(state: SessionState) -> Optional[int]:
    """
    Get the next question index that needs term extraction.
    
    Args:
        state: Current state.
        
    Returns:
        Next question index to process or None if the queue is empty.
    """
    if not state.term_extraction_queue:
        logger.debug("State: get_next_extraction_task: Extraction queue is empty.")
        return None
    
    next_index = state.term_extraction_queue[0]
    logger.debug(f"State: get_next_extraction_task: Next index to extract is {next_index}.")
    return next_index

def mark_extraction_complete(state: SessionState, question_index: int) -> SessionState:
    """
    Mark a question as having completed term extraction.
    
    Args:
        state: Current state.
        question_index: Index of the question that was processed.
        
    Returns:
        Updated SessionState.
    """
    logger.debug(f"State: mark_extraction_complete: Marking index {question_index} as complete.")
    
    # Create a new queue without the processed index
    if question_index in state.term_extraction_queue:
        new_queue = [idx for idx in state.term_extraction_queue if idx != question_index]
        logger.debug(f"State: mark_extraction_complete: Removed index {question_index} from extraction queue. "
                   f"Current queue: {new_queue}")
        return state.update(term_extraction_queue=new_queue)
    else:
        logger.debug(f"State: mark_extraction_complete: Index {question_index} not found in extraction queue: "
                   f"{state.term_extraction_queue}")
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
            "verified_answers": state.verified_answers,
            "extracted_terms": state.extracted_terms,
            "term_extraction_queue": state.term_extraction_queue,
            "current_question": get_formatted_current_question(state)
        }
        
        # Add error if present
        if state.error:
            raw_dict["error"] = state.error
        
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
            "error": f"Error converting state to dictionary: {str(e)}"
        }

# For backward compatibility with existing code
# This allows us to gradually migrate without breaking existing code
def create_updated_state(state: SessionState, **updates) -> SessionState:
    """
    Legacy compatibility function to create a new state instance with updates applied.
    
    This function calls state.update() to ensure immutability is maintained.
    
    Args:
        state: Current state to be updated
        **updates: Keyword arguments with fields to update
        
    Returns:
        A new SessionState instance with updates applied
    """
    return state.update(**updates)