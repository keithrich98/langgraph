# state.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ChatState:
    """
    Shared state for the medical questionnaire multi-agent system.
    This lightweight state class is designed to work well with the Functional API.
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
    
    # Storage for verified answers ready for term extraction
    # Format: {question_index: {"question": str, "answer": str, "verification": str}}
    verified_answers: Dict[int, Dict[str, str]] = field(default_factory=dict)
    
    # Queue of question indices ready for term extraction
    term_extraction_queue: List[int] = field(default_factory=list)
    
    # Storage for extracted medical terms
    # Format: {question_index: [term1, term2, ...]}
    extracted_terms: Dict[int, List[str]] = field(default_factory=dict)
    
    # Last processed index for term extraction
    last_extracted_index: int = -1

# Helper functions for managing the state
def add_to_extraction_queue(state: ChatState, question_index: int) -> ChatState:
    """
    Add a question index to the term extraction queue after verification.
    
    Args:
        state: Current state
        question_index: Index of the question that was verified
        
    Returns:
        Updated ChatState
    """
    if question_index not in state.term_extraction_queue:
        state.term_extraction_queue.append(question_index)
    return state

def get_next_extraction_task(state: ChatState) -> Optional[int]:
    """
    Get the next question index that needs term extraction.
    
    Args:
        state: Current state
        
    Returns:
        Next question index to process or None if queue is empty
    """
    if not state.term_extraction_queue:
        return None
    return state.term_extraction_queue[0]

def mark_extraction_complete(state: ChatState, question_index: int) -> ChatState:
    """
    Mark a question as having completed term extraction.
    
    Args:
        state: Current state
        question_index: Index of the question that was processed
        
    Returns:
        Updated ChatState
    """
    if question_index in state.term_extraction_queue:
        state.term_extraction_queue.remove(question_index)
    state.last_extracted_index = question_index
    return state