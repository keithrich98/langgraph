"""Centralized state management for the questionnaire application built with LangGraph.

This module defines the core ChatState and a helper function to initialize a new session’s state.
All nodes in the graph (e.g. in parent_workflow.py, answer_verifier.py, question_processor.py, term_extractor.py)
should import ChatState (and init_state) from this module to ensure consistency.
"""

from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    current_question_index: int
    questions: List[Dict]
    conversation_history: Annotated[List, add_messages]
    responses: Dict[int, str]
    is_complete: bool
    verified_answers: Dict[int, Dict[str, str]]
    term_extraction_queue: List[int]
    extracted_terms: Dict[int, List[str]]
    last_extracted_index: int
    verification_result: Dict[str, Any]
    thread_id: Optional[str]
    trigger_extraction: bool

def init_state(thread_id: Optional[str] = None) -> ChatState:
    """Returns an initial ChatState for a new questionnaire session.
    
    An optional thread_id can be provided if resuming from a particular session.
    """
    return ChatState(
        current_question_index=0,
        questions=[],
        conversation_history=[],  # To be updated with messages (using the add_messages reducer)
        responses={},
        is_complete=False,
        verified_answers={},
        term_extraction_queue=[],
        extracted_terms={},
        last_extracted_index=-1,
        verification_result={},
        thread_id=thread_id,
        trigger_extraction=False,
    )
