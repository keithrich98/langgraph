from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, Any
import uvicorn
import os

# Import graph and state from parent_workflow, and shared utilities.
from parent_workflow import graph, ChatState, init_node
from langgraph.types import Command
from logging_config import logger
from utils import convert_messages, get_logging_context
from state import ChatState, init_state
from fastapi.responses import JSONResponse

# FastAPI application initialization.
app = FastAPI()

# Configure CORS for frontend connectivity.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if we are in DEBUG mode (adjust via env variable or config).
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API response models.
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

# Helper to sync state with shared memory.
def sync_state_to_memory_saver(thread_id: str, state_values: Dict):
    from parent_workflow import shared_memory
    try:
        logger.debug(f"Syncing state with thread_id {thread_id} to memory saver")
        import copy
        state_copy = copy.deepcopy(state_values)
        shared_memory.save(thread_id, state_copy)
        test_load = shared_memory.load(thread_id)
        if test_load:
            queue = test_load.get('term_extraction_queue', [])
            terms = test_load.get('extracted_terms', {})
            logger.debug(f"State sync verified - Queue: {queue}, Terms keys: {list(terms.keys())}")
        else:
            logger.warning("State sync verification failed - loaded state is empty")
    except Exception as e:
        logger.error(f"Error syncing state to memory saver: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def get_current_question(conversation_history: List) -> str:
    # Uses the helper imported from message_utils
    dict_history = convert_messages(conversation_history)
    for msg in reversed(dict_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""

# --- /start endpoint ---
@app.post("/start", response_model=StartResponse)
async def start_session():
    """
    Start a new questionnaire session by generating a unique thread ID,
    initializing minimal state, and invoking the graph to get the first question.
    """
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        logger.debug(f"Starting new session with thread_id: {thread_id}")
        initial_state: ChatState = init_state(thread_id)
        logger.debug(f"Created initial state: {initial_state}")
        
        # Invoke graph until it hits the first interrupt.
        result = graph.invoke(initial_state, config)
        logger.debug(f"Graph invoke result type: {type(result)}")
        logger.debug(f"Graph invoke result: {result}")
        
        # Retrieve and sync state.
        current_state = graph.get_state(config)
        logger.debug("Getting current state")
        logger.debug(f"Current state values keys: {current_state.values.keys() if hasattr(current_state.values, 'keys') else 'N/A'}")
        
        sync_state_to_memory_saver(thread_id, current_state.values)
        
        waiting_for_input = False
        if current_state.tasks and len(current_state.tasks) > 0:
            task = current_state.tasks[0]
            logger.debug(f"Task: {task}")
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        
        state_values = current_state.values
        logger.debug(f"State values: {state_values}")
        
        raw_conversation_history = []
        if isinstance(state_values, dict):
            logger.debug("State values is a dict, using get method")
            raw_conversation_history = state_values.get('conversation_history', [])
        elif hasattr(state_values, 'conversation_history'):
            logger.debug("State values has conversation_history attribute")
            raw_conversation_history = state_values.conversation_history
        else:
            logger.debug("Couldn't find conversation_history, using empty list")
        
        conversation_history = convert_messages(raw_conversation_history)
        logger.debug(f"Converted conversation history: {conversation_history}")
        
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
        msg = f"Error starting session: {str(e)}"
        logger.error(msg)
        if DEBUG:
            return JSONResponse(status_code=500, content={"detail": msg, "trace": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=msg)

# --- /next endpoint ---
@app.post("/next", response_model=NextResponse)
async def next_step(req: NextRequest):
    """
    Process the user's answer and return the next question.
    """
    logger.debug(f"Processing next step for session: {req.session_id}")
    thread_id = req.session_id
    config = {"configurable": {"thread_id": thread_id}}
    try:
        current_state = graph.get_state(config)
        state_values = current_state.values
        
        has_thread_id = state_values.get('thread_id') is not None
        
        if not has_thread_id:
            logger.debug(f"Adding thread_id {thread_id} to state before invoking graph")
            update_cmd = {"thread_id": thread_id}
            graph.invoke(update_cmd, config)
            updated_after_thread_id = graph.get_state(config)
            sync_state_to_memory_saver(thread_id, updated_after_thread_id.values)
        
        graph.invoke(Command(resume=req.answer), config)
        
        updated_state = graph.get_state(config)
        state_values = updated_state.values
        sync_state_to_memory_saver(thread_id, state_values)
        
        waiting_for_input = False
        if updated_state.tasks and len(updated_state.tasks) > 0:
            task = updated_state.tasks[0]
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        finished = not waiting_for_input and not updated_state.next
        
        raw_conversation_history = state_values.get('conversation_history', [])
        conversation_history = convert_messages(raw_conversation_history)
        current_question = get_current_question(raw_conversation_history)
        
        logger.debug(f"Next step processed with thread_id: {thread_id}")
        logger.debug(f"Thread ID stored in state: {state_values.get('thread_id')}")
        logger.debug(f"Waiting for input: {waiting_for_input}, finished: {finished}")
        logger.debug(f"Current term_extraction_queue: {state_values.get('term_extraction_queue', [])}")
        logger.debug(f"Current extracted_terms keys: {list(state_values.get('extracted_terms', {}).keys())}")
        logger.debug(f"Current state keys: {list(state_values.keys())}")
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=finished,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        import traceback
        msg = f"Error processing answer: {str(e)}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        if DEBUG:
            return JSONResponse(status_code=500, content={"detail": msg, "trace": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=msg)

# --- /terms endpoint ---
class TermsResponse(BaseModel):
    """Response model for terms extraction."""
    terms: Dict[int, List[str]]
    pending_extraction: List[int]

@app.get("/terms/{session_id}", response_model=TermsResponse)
async def get_extracted_terms(session_id: str):
    """
    Retrieve extracted terms for a session.
    This endpoint returns the current state without triggering extraction;
    extraction is handled asynchronously in the background.
    """
    logger.debug(f"Retrieving extracted terms for session: {session_id}")
    config = {"configurable": {"thread_id": session_id}}
    try:
        from parent_workflow import shared_memory
        
        raw_states = getattr(shared_memory, "_states", {})
        if session_id in raw_states:
            logger.debug("DEBUG: Direct memory saver access - session_id found", extra={"session_id": session_id})
            direct_state = raw_states[session_id]
            if direct_state and 'extracted_terms' in direct_state:
                direct_terms = direct_state['extracted_terms']
                logger.debug(f"DEBUG: Direct extracted_terms: {direct_terms}", extra={"session_id": session_id})
                logger.debug(f"DEBUG: Direct terms keys: {list(direct_terms.keys())}", extra={"session_id": session_id})
        else:
            logger.debug("DEBUG: Direct memory saver access - session_id NOT found", extra={"session_id": session_id})
            logger.debug(f"DEBUG: Available session IDs: {list(raw_states.keys())}", extra={"session_id": session_id})
        
        memory_state = shared_memory.load(session_id)
        if memory_state and 'extracted_terms' in memory_state:
            logger.debug(f"Using state from memory_saver for session {session_id}", extra={"session_id": session_id})
            extracted_terms = memory_state.get('extracted_terms', {})
            term_extraction_queue = memory_state.get('term_extraction_queue', [])
            logger.debug(f"Found extracted_terms: {extracted_terms}", extra={"session_id": session_id})
            logger.debug(f"Found extracted_terms keys: {list(extracted_terms.keys())}", extra={"session_id": session_id})
            for k, v in extracted_terms.items():
                logger.debug(f"Terms for index {k}: {v[:3] if len(v) > 3 else v} (total: {len(v)})", extra={"session_id": session_id})
            logger.debug(f"All keys in memory state: {list(memory_state.keys())}", extra={"session_id": session_id})
            if 'last_extracted_index' in memory_state:
                logger.debug(f"last_extracted_index: {memory_state['last_extracted_index']}", extra={"session_id": session_id})
        else:
            logger.warning(f"No valid state found in memory_saver for session {session_id}", extra={"session_id": session_id})
            current_state = graph.get_state(config)
            state_values = current_state.values
            extracted_terms = state_values.get('extracted_terms', {})
            term_extraction_queue = state_values.get('term_extraction_queue', [])
            sync_state_to_memory_saver(session_id, state_values)
            memory_state = shared_memory.load(session_id)
            if memory_state and 'extracted_terms' in memory_state:
                logger.debug(f"After sync, extracted_terms: {memory_state.get('extracted_terms', {})}", extra={"session_id": session_id})
        
        if hasattr(shared_memory, "_states") and session_id in shared_memory._states:
            direct_extracted_terms = shared_memory._states[session_id].get('extracted_terms', {})
            if direct_extracted_terms:
                logger.debug(f"Using direct access - extracted_terms keys: {list(direct_extracted_terms.keys())}", extra={"session_id": session_id})
                extracted_terms = direct_extracted_terms
        
        normalized_terms = {str(k): v for k, v in extracted_terms.items()}
        try:
            normalized_queue = [int(item) for item in term_extraction_queue]
        except (TypeError, ValueError):
            logger.warning(f"Could not convert extraction queue items to integers: {term_extraction_queue}", extra={"session_id": session_id})
            normalized_queue = term_extraction_queue
        
        logger.info(f"Current extraction status - Queue: {normalized_queue}, Extracted terms: {list(normalized_terms.keys())}", extra={"session_id": session_id})
        logger.info(f"Terms content sample: {next(iter(normalized_terms.values()), [])[:3] if normalized_terms else []}", extra={"session_id": session_id})
        
        return TermsResponse(
            terms=normalized_terms,
            pending_extraction=normalized_queue
        )
    except Exception as e:
        import traceback
        msg = f"Error retrieving extracted terms: {str(e)}"
        logger.error(msg)
        logger.error(traceback.format_exc())
        if DEBUG:
            return JSONResponse(status_code=500, content={"detail": msg, "trace": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)