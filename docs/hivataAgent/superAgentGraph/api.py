# api.py with HIL API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, Any
import uvicorn

# Import our graph and ChatState from parent_workflow.py
from parent_workflow import graph, ChatState, init_node
from langgraph.types import Command  # Import Command for resuming execution
from logging_config import logger

app = FastAPI()

# Configure CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartResponse(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_question: str
    waiting_for_input: bool

class NextRequest(BaseModel):
    session_id: str
    answer: str

class NextResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    current_question: Optional[str] = None
    finished: bool
    waiting_for_input: bool

def convert_messages_to_dict(messages):
    """
    Convert LangChain message objects to dictionaries.
    
    Ensures consistent conversion from various message formats to a standardized dict format.
    Handles both objects with attributes and dictionary-style messages.
    
    Args:
        messages: A list of message objects (AIMessage, HumanMessage, dict, etc.)
        
    Returns:
        list: A list of dicts with 'role' and 'content' keys
    """
    result = []
    if not messages:
        return result
        
    for msg in messages:
        # Handle dict-style messages
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            # Already in the right format
            result.append(msg.copy())  # Use copy to avoid mutating original
            continue
            
        # Handle message objects with attributes
        if hasattr(msg, 'content'):
            content = msg.content
            
            # Determine the role with fallbacks
            if hasattr(msg, 'type'):
                role = msg.type.lower()
            elif hasattr(msg, '_message_type'):
                msg_type = msg._message_type.lower() if hasattr(msg, '_message_type') else ''
                if 'ai' in msg_type or 'assistant' in msg_type:
                    role = 'ai'
                elif 'human' in msg_type or 'user' in msg_type:
                    role = 'human'
                else:
                    role = 'ai'  # Default fallback
            else:
                # Try to infer from class name
                class_name = msg.__class__.__name__.lower()
                if 'ai' in class_name or 'assistant' in class_name:
                    role = 'ai'
                elif 'human' in class_name or 'user' in class_name:
                    role = 'human'
                else:
                    role = 'ai'  # Default fallback
                    
            result.append({"role": role, "content": content})
            continue
            
        # Log any unhandled message types
        logger.warning(f"Unhandled message type: {type(msg)}")
        
    logger.debug(f"Converted {len(messages)} messages to {len(result)} dict messages")
    return result

def get_current_question(conversation_history: List) -> str:
    """Extract the latest AI message from the conversation history."""
    # First convert any message objects to dictionaries
    dict_history = convert_messages_to_dict(conversation_history)
    
    # Then extract the last AI message
    for msg in reversed(dict_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    # Generate a unique thread ID
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        logger.debug(f"Starting new session with thread_id: {thread_id}")
        
        # Initialize with minimal required state
        initial_state = {
            "current_question_index": 0,
            "questions": [],
            "conversation_history": [],
            "responses": {},
            "is_complete": False,
            "verified_answers": {},
            "term_extraction_queue": [],
            "extracted_terms": {},
            "last_extracted_index": -1,
            "verification_result": {},
            # Include thread_id in state for term extraction
            "thread_id": thread_id,
            "trigger_extraction": False
        }
        logger.debug(f"Created initial state: {initial_state}")
        
        # Run the graph until it hits the first interrupt
        logger.debug("About to invoke graph")
        result = graph.invoke(initial_state, config)
        logger.debug(f"Graph invoke result type: {type(result)}")
        logger.debug(f"Graph invoke result: {result}")
        
        # Get the current state
        logger.debug("Getting current state")
        current_state = graph.get_state(config)
        logger.debug(f"Current state type: {type(current_state)}")
        logger.debug(f"Current state attributes: {dir(current_state)}")
        logger.debug(f"Current state values type: {type(current_state.values)}")
        logger.debug(f"Current state values keys: {current_state.values.keys() if hasattr(current_state.values, 'keys') else 'No keys method'}")
        
        # IMPORTANT: Explicitly save the state to our memory saver to ensure background
        # processes can access it
        sync_state_to_memory_saver(thread_id, current_state.values)
        
        # Check if waiting for input
        waiting_for_input = False
        if current_state.tasks and len(current_state.tasks) > 0:
            task = current_state.tasks[0]
            logger.debug(f"Task: {task}")
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        
        # Get conversation history
        state_values = current_state.values
        logger.debug(f"State values: {state_values}")
        
        # Try different ways to access conversation history
        raw_conversation_history = []
        if isinstance(state_values, dict):
            logger.debug("State values is a dict, using get method")
            raw_conversation_history = state_values.get('conversation_history', [])
        elif hasattr(state_values, 'conversation_history'):
            logger.debug("State values has conversation_history attribute")
            raw_conversation_history = state_values.conversation_history
        else:
            logger.debug("Couldn't find conversation_history, using empty list")
        
        # Convert message objects to dictionaries
        conversation_history = convert_messages_to_dict(raw_conversation_history)
        logger.debug(f"Converted conversation history: {conversation_history}")
        
        # Get the current question
        current_question = get_current_question(raw_conversation_history)
        
        logger.debug(f"Session started successfully, conversation history length: {len(conversation_history)}")
        
        return StartResponse(
            session_id=thread_id,
            conversation_history=conversation_history,
            current_question=current_question,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        import traceback
        logger.error(f"Error starting session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question."""
    logger.debug(f"Processing next step for session: {req.session_id}")
    thread_id = req.session_id
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get the current state to update the thread_id
        current_state = graph.get_state(config)
        state_values = current_state.values
        
        # Check if thread_id is already in the state
        has_thread_id = state_values.get('thread_id') is not None
        
        # If thread_id isn't in state, add it before continuing
        if not has_thread_id:
            logger.debug(f"Adding thread_id {thread_id} to state before invoking graph")
            # We need to include the thread_id in the state for background term extraction
            # Create a command that will update the state with the thread_id
            update_cmd = {"thread_id": thread_id}
            # Apply the update before resuming
            graph.invoke(update_cmd, config)
            
            # Get the updated state and explicitly save it to our memory saver
            updated_after_thread_id = graph.get_state(config)
            sync_state_to_memory_saver(thread_id, updated_after_thread_id.values)
        
        # Resume execution with the user's answer
        graph.invoke(Command(resume=req.answer), config)
        
        # Get the updated state
        updated_state = graph.get_state(config)
        
        # IMPORTANT: Explicitly save the state to our memory saver to ensure background
        # processes can access it
        state_values = updated_state.values
        sync_state_to_memory_saver(thread_id, state_values)
        
        # Check if waiting for input
        waiting_for_input = False
        if updated_state.tasks and len(updated_state.tasks) > 0:
            task = updated_state.tasks[0]
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        
        # Check if finished
        finished = not waiting_for_input and not updated_state.next
        
        # Get conversation history and current question
        # Use get() method since state_values is a dictionary
        raw_conversation_history = state_values.get('conversation_history', [])
        
        # Convert message objects to dictionaries
        conversation_history = convert_messages_to_dict(raw_conversation_history)
        
        current_question = get_current_question(raw_conversation_history)
        
        # Debug log additional information for troubleshooting
        logger.debug(f"Next step processed with thread_id: {thread_id}")
        logger.debug(f"Thread ID stored in state: {state_values.get('thread_id')}")
        logger.debug(f"Waiting for input: {waiting_for_input}, finished: {finished}")
        logger.debug(f"Current term_extraction_queue: {state_values.get('term_extraction_queue', [])}")
        logger.debug(f"Current extracted_terms: {list(state_values.get('extracted_terms', {}).keys())}")
        logger.debug(f"Current state keys: {list(state_values.keys())}")
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=finished,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        import traceback
        logger.error(f"Error processing answer: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

def sync_state_to_memory_saver(thread_id, state_values):
    """
    Ensure the state is explicitly saved to our memory saver.
    This bridges the gap between LangGraph's internal state and our custom memory saver.
    """
    from parent_workflow import shared_memory
    
    try:
        logger.debug(f"Syncing state with thread_id {thread_id} to memory saver")
        
        import copy
        state_copy = copy.deepcopy(state_values)  # changed from dict comprehension to deep copy
        
        shared_memory.save(thread_id, state_copy)
        
        test_load = shared_memory.load(thread_id)
        if test_load:
            queue = test_load.get('term_extraction_queue', [])
            terms = test_load.get('extracted_terms', {})
            logger.debug(f"State sync verified - Queue: {queue}, Terms: {list(terms.keys())}")
        else:
            logger.warning("State sync verification failed - loaded state is empty")
    except Exception as e:
        logger.error(f"Error syncing state to memory saver: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
class TermsResponse(BaseModel):
    """Response model for terms extraction."""
    terms: Dict[int, List[str]]
    pending_extraction: List[int]
    
@app.get("/terms/{session_id}", response_model=TermsResponse)
def get_extracted_terms(session_id: str):
    """
    Retrieve extracted terms for a session.
    This endpoint simply returns the current state without triggering extraction.
    Extraction is now handled automatically in the background via threading.
    """
    logger.debug(f"Retrieving extracted terms for session: {session_id}")
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get direct access to the memory saver to debug internal state
        from parent_workflow import shared_memory
        
        # Debug the internal state of memory_saver
        raw_states = getattr(shared_memory, "_states", {})
        if session_id in raw_states:
            logger.debug(f"DEBUG: Direct memory saver access - session_id found in _states")
            direct_state = raw_states[session_id]
            if direct_state and 'extracted_terms' in direct_state:
                direct_terms = direct_state['extracted_terms']
                logger.debug(f"DEBUG: Direct extracted_terms: {direct_terms}")
                logger.debug(f"DEBUG: Direct terms keys: {list(direct_terms.keys())}")
        else:
            logger.debug(f"DEBUG: Direct memory saver access - session_id NOT found in _states")
            logger.debug(f"DEBUG: Available session IDs in memory saver: {list(raw_states.keys())}")
        
        # Try to load state from memory saver
        memory_state = shared_memory.load(session_id)
        
        # If memory_state exists and has extracted terms, use that data
        if memory_state and 'extracted_terms' in memory_state:
            logger.debug(f"Using state from memory_saver for session {session_id}")
            extracted_terms = memory_state.get('extracted_terms', {})
            term_extraction_queue = memory_state.get('term_extraction_queue', [])
            
            # Log detailed information about extracted terms
            logger.debug(f"Found extracted_terms dict: {extracted_terms}")
            if extracted_terms:
                logger.debug(f"Found extracted terms with keys: {list(extracted_terms.keys())}")
                for k, v in extracted_terms.items():
                    logger.debug(f"Terms for index {k}: {v[:3] if len(v) > 3 else v}... (total: {len(v)})")
            else:
                logger.warning(f"extracted_terms is empty or not properly formatted: {extracted_terms}")
                
            # Debug all keys in the state for troubleshooting
            logger.debug(f"All keys in memory state: {list(memory_state.keys())}")
            if 'last_extracted_index' in memory_state:
                logger.debug(f"last_extracted_index: {memory_state['last_extracted_index']}")
        else:
            logger.warning(f"No valid state found in memory_saver for session {session_id}")
            # Fall back to getting state from graph
            current_state = graph.get_state(config)
            state_values = current_state.values
            
            # Extract the terms and pending extraction queue
            extracted_terms = state_values.get('extracted_terms', {})
            term_extraction_queue = state_values.get('term_extraction_queue', [])
            
            # Sync the state to our memory saver for future use
            sync_state_to_memory_saver(session_id, state_values)
            
            # After syncing, try to get the direct state again for verification
            memory_state = shared_memory.load(session_id)
            if memory_state and 'extracted_terms' in memory_state:
                logger.debug(f"After sync, extracted_terms: {memory_state.get('extracted_terms', {})}")
        
        # Direct access to internal state for debugging
        if hasattr(shared_memory, "_states") and session_id in shared_memory._states:
            direct_extracted_terms = shared_memory._states[session_id].get('extracted_terms', {})
            if direct_extracted_terms:
                logger.debug(f"Using direct access to _states - extracted_terms found with keys: {list(direct_extracted_terms.keys())}")
                # Use the direct access results if available
                extracted_terms = direct_extracted_terms
        
        # Ensure all keys in extracted_terms are strings (for JSON compatibility)
        normalized_terms = {}
        for k, v in extracted_terms.items():
            normalized_terms[str(k)] = v
        
        # Ensure term_extraction_queue items are integers
        try:
            normalized_queue = [int(item) for item in term_extraction_queue]
        except (TypeError, ValueError):
            logger.warning(f"Could not convert all queue items to integers, using as-is: {term_extraction_queue}")
            normalized_queue = term_extraction_queue
        
        logger.info(f"Current extraction status - Queue: {normalized_queue}, Extracted terms: {list(normalized_terms.keys())}")
        logger.info(f"Terms content sample: {next(iter(normalized_terms.values()), [])[:3] if normalized_terms else []}")
        
        # Just return the current state - extraction happens automatically in the background
        return TermsResponse(
            terms=normalized_terms,
            pending_extraction=normalized_queue
        )
    except Exception as e:
        logger.error(f"Error retrieving extracted terms: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving extracted terms: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)