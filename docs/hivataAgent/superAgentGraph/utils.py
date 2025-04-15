# utils.py
"""
General utility functions used throughout the application.

This module provides:
  - Message conversion helpers for converting various message formats
    into a standardized dictionary format.
  - A logging context helper (get_logging_context) to extract common fields
    (such as thread_id and current_question_index) from the ChatState.
"""

from typing import Dict, List, Literal, Any, Optional
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict  # Corrected import
from langchain_core.messages import AIMessage, HumanMessage
from logging_config import logger

# --- Define the ChatState Schema ---
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

def get_message_role(m: Any) -> str:
    """
    Determine the role of a message object based on its attributes.

    Priority:
      1. Use the 'type' attribute if available.
      2. Otherwise, use the '_message_type' attribute.
      3. Otherwise, infer from the class name.

    Returns:
      A lowercase string representing the role ('ai' or 'human').
    """
    if hasattr(m, 'type'):
        return m.type.lower()
    elif hasattr(m, '_message_type'):
        msg_type = m._message_type.lower()
        return 'ai' if 'ai' in msg_type or 'assistant' in msg_type else 'human'
    else:
        class_name = m.__class__.__name__.lower()
        return 'ai' if 'ai' in class_name or 'assistant' in class_name else 'human'

def convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert a list of message objects or dictionaries into a list of dictionaries 
    with keys 'role' and 'content'.

    Args:
        messages: A list of message objects (e.g., AIMessage, HumanMessage) or dictionaries.

    Returns:
        A list of dictionaries with standardized keys.
    """
    logger.debug(f"Converting {len(messages)} message(s) to dict format.", extra={"source_module": "utils"})
    result = []
    for m in messages:
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            result.append(m.copy())
        else:
            try:
                if hasattr(m, 'content'):
                    content = m.content
                    role = get_message_role(m)
                    result.append({"role": role, "content": content})
                else:
                    logger.warning(f"convert_messages: Could not convert message: {m}", extra={"source_module": "utils"})
            except Exception as e:
                logger.error(f"convert_messages: Error converting message: {str(e)}", extra={"source_module": "utils", "message": str(m)})
    logger.debug(f"Converted messages: {result}", extra={"source_module": "utils"})
    return result

def get_logging_context(state: ChatState, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Returns a dictionary of logging context extracted from the current state.

    Args:
        state: The current ChatState.
        extra: Optional dictionary of additional keys to add.

    Returns:
        A dictionary with keys 'thread_id' and 'question_index'.
    """
    context = {
        "thread_id": state.get("thread_id"),
        "question_index": state.get("current_question_index")
    }
    if extra:
        context.update(extra)
    return context

def format_question_prompt(question_obj: Dict[str, Any]) -> str:
    """
    Formats a question object by combining its text and requirements
    into a prompt suitable for an LLM.
    
    Args:
        question_obj: A dictionary containing the question text and requirements.
        
    Returns:
        A formatted string prompt.
    """
    formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in question_obj.get("requirements", {}).items()])
    return f"{question_obj.get('text', '')}\n\nRequirements:\n{formatted_requirements}"